# Text-MedSAM: Text-Guided Medical Image Segmentation

A text-guided medical image segmentation framework built on the MedSAM architecture. This repository implements medical image segmentation using natural language prompts combined with efficient vision transformers and biomedical text encoders.

## Overview

Text-MedSAM extends the Segment Anything Model (SAM) for medical imaging by incorporating text prompts as guidance for segmentation tasks. The framework leverages:

- **Efficient Image Encoding**: RepViT-based encoder for fast visual feature extraction
- **Medical Text Encoding**: PubMedBERT for domain-specific biomedical text understanding
- **Adaptive Text-Image Fusion**: Token shuffling and channel unshuffling for seamless modality integration
- **Multi-Modal Attention**: Cross-modal attention mechanisms for text-guided segmentation
- **Distributed Training**: DDP support for efficient large-scale model training

### Key Features

- ✅ Text-guided segmentation of medical images (CT, MRI, PET, etc.)
- ✅ Support for both 2D and 3D medical imaging data
- ✅ Distributed Data Parallel (DDP) training on multi-GPU systems
- ✅ Knowledge distillation from teacher models
- ✅ Efficient inference with optimized model architecture
- ✅ Comprehensive data preprocessing utilities
- ✅ Medical-domain text embeddings with PubMedBERT

---

## System Requirements

```
OS: Ubuntu 20.04+
Python: 3.9+
CUDA: 12.2+
PyTorch: 2.0+
GPU Memory: 8GB minimum (for inference), 24GB+ recommended (for training)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/muxin-wei/Text-MedSAM.git
   cd Text-MedSAM
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Download Pre-trained Models**
   ```bash
   mkdir -p ckpts
   # Download PubMedBERT and place in ckpts/
   ```

4. **Docker Setup** (Optional)
   ```bash
   docker build -t text-medsam .
   docker run -it --gpus all -v /path/to/data:/inputs -v /path/to/outputs:/outputs text-medsam
   ```

---

## Repository Structure

```
Text-MedSAM/
├── src/
│   ├── dataset/
│   │   ├── textseg.py            # Text-guided segmentation dataset
│   │   └── utils.py              # Data processing utilities
│   ├── models/
│   │   ├── shuffle_former.py     # ShuffleFormer: Text-image fusion architecture
│   │   ├── transformer.py        # Attention mechanisms
│   │   ├── modules.py            # Core building blocks
│   │   ├── visual/
│   │   │   └── image_encoder.py  # RepViT encoder
│   │   ├── heads/
│   │   │   └── text_embedder.py  # PubMedBERT text encoder
│   │   └── maskformer.py         # MaskFormer decoder
│   └── losses/
│       ├── total_loss.py         # Combined loss function
│       ├── medsam_loss.py        # Segmentation loss
│       └── clip_loss.py          # Text-image alignment loss
├── configs/
│   └── text_seg_repvit.yaml      # Training configuration
├── scripts/
│   ├── embeddings.sh             # DDP embedding generation
│   ├── text_predict.sh           # Batch inference
│   └── batch_infer_text.py       # Inference implementation
├── utils/
│   ├── utils.py                  # Training utilities
│   └── npz_to_npy.py            # Data format conversion
├── ckpts/                         # Model checkpoints
├── main.py                        # Lightning training framework
├── train_one_gpu.py              # Single GPU training
├── sample_embedding.py            # DDP embedding computation
└── requirements.txt              # Dependencies
```

---

## Model Architecture

### System Overview

```
Medical Image + Text Prompt
    ↓
┌─────────────────────────────────────────┐
│     MULTI-MODAL ENCODING STAGE          │
├─────────────────────────────────────────┤
│  Image Encoder (RepViT)                 │
│  - Input: 256×256 RGB image             │
│  - Output: 64×64×256 feature map        │
│                  ↓                      │
│  Text Encoder (PubMedBERT)              │
│  - Input: Medical text prompt           │
│  - Output: 1×1024 embedding             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   TEXT-IMAGE FUSION & ATTENTION STAGE   │
│      (ShuffleFormer Architecture)       │
├─────────────────────────────────────────┤
│  Token Shuffling                        │
│  - Spatial rearrangement of tokens      │
│  - Channel dimension reduction          │
│  - Enables local token interaction      │
│                  ↓                      │
│  Transformer Blocks with Fusion         │
│  - Self-attention on shuffled tokens    │
│  - Cross-modal text-image attention     │
│  - Multi-head attention (8 heads)       │
│  - LayerNorm + MLP blocks               │
│                  ↓                      │
│  Channel Unshuffling                    │
│  - Restore spatial organization         │
│  - Channel expansion                    │
│  - Reconstruct feature map              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│    ATTENTION GATE & DECODING STAGE      │
├─────────────────────────────────────────┤
│  Attention Gate (AttenGate)             │
│  - Gated attention mechanism            │
│  - Fuses image features and             │
│    text-guided attention maps           │
│  - Output: Attentive feature fusion     │
│                  ↓                      │
│  Atrous Separable Convolution           │
│  - Dilated convolution (kernel=3,       │
│    dilation=2, padding=2)               │
│  - Multi-scale receptive field          │
│  - BatchNorm + GELU activation          │
└─────────────────────────────────────────┘
    ↓
Segmentation Mask (256×256)
```

---

## Component Details

### 1. **Image Encoder: RepViT**

**Purpose**: Extract multi-scale visual features from medical images

**Architecture**:
- **Input**: 256×256 medical image
- **Type**: Efficient Vision Transformer (lightweight alternative to ViT)
- **Output**: 64×64×256 feature map (16× downsampling)
- **Efficiency**:
  - Parameters: ~6M (comparable to TinyViT)
  - Inference Latency: 0.36s per 512×512 image
  - Designed for resource-constrained medical imaging environments

**Key Advantages**:
- Optimized for real-time inference
- Maintains spatial information through patch embeddings
- Learns both local and global visual context
- Suitable for portable medical imaging devices

---

### 2. **Text Encoder: PubMedBERT**

**Purpose**: Convert medical text prompts into semantic embeddings

**Architecture Details**:
- **Model**: BERT variant pre-trained on 4.5B+ PubMed tokens
- **Vocabulary**: Medical and biomedical terminology
- **Configuration**:
  - Hidden Dimension: 768
  - Number of Layers: 12
  - Attention Heads: 12
  - Output Projection: 1024 (after final linear layer)

**Text Processing Pipeline**:
```
Raw Text Input (e.g., "left kidney")
    ↓
Tokenization (max 256 tokens)
    ↓
PubMedBERT Encoding (768-dim)
    ↓
Linear Projection (768 → 1024)
    ↓
Text Embedding (1×1024)
```

**Medical Domain Knowledge**:
- Trained on biomedical abstracts and literature
- Better understanding of anatomical terms (kidney, liver, etc.)
- Captures pathological context (lesion, tumor, etc.)
- Outperforms general-purpose BERT on medical tasks

**Text Prompt Format**:
```json
{
  "CT_Abd": {
    "1": ["kidney", "left kidney", "renal cortex"],
    "2": ["liver", "hepatic parenchyma"],
    "3": ["spleen"],
    "instance_label": 0
  }
}
```

---

### 3. **Text-Image Fusion: ShuffleFormer**

**Purpose**: Seamlessly integrate visual and textual information for text-guided segmentation

**Architecture Overview**:

The ShuffleFormer implements a sophisticated fusion strategy combining token shuffling, cross-modal attention, and learnable fusion:

#### **A. Token Shuffling (TokenShuffle)**

**Goal**: Reorganize spatial tokens to enable efficient multi-scale fusion

**Process**:
```
Input: Visual Features (B × 4096 × 256)  [64×64 flattened]
       where B = batch size

Step 1: MLP Projection
    ├─ Linear: 256 → 256 // (s²)
    │  [where s = shuffle_size, typically 2]
    │  Output: (B × 4096 × 64)
    └─ Creates reduced dimension for efficiency

Step 2: Spatial Rearrangement
    ├─ Input shape: (B × 4096 × 64)
    ├─ Rearrange to: (B × 1024 × 256)  [32×32 with 4× channels]
    ├─ Formula: "b (h s1 w s2) c -> b (h w) (c s1 s2)"
    │  where h=32, w=32, s1=2, s2=2
    └─ Aggregates local spatial information

Step 3: Fusion MLP (Optional)
    ├─ For each fusion block:
    │  └─ Linear: 256 → 256
    └─ Refines shuffled representation

Output: Fused Token Representation (B × 1024 × 256)
```

**Benefits**:
- Reduces computational complexity
- Groups spatially proximal tokens for local interaction
- Enables multi-scale feature fusion

---

#### **B. Transformer Blocks with Cross-Modal Attention**

**Goal**: Learn text-guided visual features through multi-head attention

**Architecture** (repeats `n_attn` times, typically 2):
```
Input: Visual Tokens (B × 1024 × 256)
       Text Embedding (B × 1 × 1024)

Step 1: Token Concatenation
    ├─ Concatenate along sequence dimension
    ├─ Combined: (B × 1025 × combined_dim)
    └─ Text acts as guidance signal

Step 2: LayerNorm + Self-Attention
    ├─ Normalize: RMSNorm (B × 1025 × 256)
    ├─ Attention: Multi-head self-attention
    │  ├─ Query, Key, Value from same input
    │  ├─ 8 attention heads
    │  ├─ Learns text-guided visual correlations
    │  └─ Output: (B × 1025 × 256)
    └─ Residual connection: x_out = x + Attention(x)

Step 3: Text Token Extraction & Update
    ├─ Extract first token: new_text = x[0]  (1×256)
    ├─ EMA Update: text = α·text + (1-α)·new_text
    │  where α = exp(text_ema).clamp(0, 0.999)
    │  Progressive refinement of text representation
    └─ Remove text from sequence for next layer

Step 4: LayerNorm + MLP
    ├─ Normalize: RMSNorm
    ├─ MLP expansion: 256 → 1024 → 256
    ├─ GELU activation
    └─ Residual: x_out = x + MLP(x)

Output: Text-Guided Visual Features (B × 1024 × 256)
        Updated Text Embedding (B × 1 × 1024)
```

**Cross-Modal Fusion Mechanism**:
- Text embedding serves as *positional guidance* in attention
- Visual tokens are conditioned on text semantics
- Bidirectional information flow: visual features influence text refinement
- Progressive text update via EMA ensures stability

---

#### **C. Channel Unshuffling (ChannelUnshuffle)**

**Goal**: Restore spatial resolution and reconstruct full-dimensional features

**Process**:
```
Input: Fused Features (B × 1024 × 256)

Step 1: Projection
    ├─ MLP: 256 → 256
    └─ Output: (B × 1024 × 256)

Step 2: Spatial Expansion
    ├─ Rearrange: "b (h w) (c s1 s2) -> b (h s1 w s2) c"
    ├─ From (B × 1024 × 256) → (B × 4096 × 64)
    ├─ Formula expands: 32×32×256 → 64×64×64
    └─ Restores original spatial resolution

Step 3: Inverse Fusion (Optional)
    ├─ For each fusion block:
    │  └─ MLP: 64 → 64
    └─ Refines unshuffled features

Step 4: Reshape to Spatial Format
    ├─ Rearrange: "b (h w) c -> b c h w"
    ├─ From (B × 4096 × 64) → (B × 64 × 64 × 64)
    └─ Restored spatial organization

Output: Full-Resolution Features (B × 64 × 64 × 64)
```

---

### 4. **Attention Gate (AttenGate)**

**Purpose**: Adaptively combine image features with text-guided attention

**Architecture**:
```
Input: 
  - Original Image Features x: (B × 448 × 64 × 64)
  - Text-Guided Features g: (B × 64 × 64 × 64)

Step 1: Channel Projection
    ├─ Project x: Conv2d(448, int_dim, 1)
    │  └─ int_dim = (448 + 64) / 2 = 256
    │  Output: (B × 256 × 64 × 64)
    └─ Project g: Conv2d(64, int_dim, 1)
       Output: (B × 256 × 64 × 64)

Step 2: Compute Attention Map
    ├─ Combine: x + g
    ├─ GELU activation
    ├─ Sigmoid attention: ψ(g) = Conv2d(256, 1, 1)
    │  Output: (B × 1 × 64 × 64)
    └─ Attention weights: α ∈ [0, 1]

Step 3: Gated Feature Fusion
    ├─ Multiply attention by original image: α * x
    │  └─ Output: (B × 448 × 64 × 64)
    ├─ Concatenate with text-guided features
    │  └─ Output: (B × 512 × 64 × 64)
    └─ Selective feature passing based on text relevance

Output: Fused Features (B × 512 × 64 × 64)
        Attention Weight Map (B × 1 × 64 × 64)
```

**Key Mechanism**:
- Learns which parts of the image are relevant to the text prompt
- Attention map highlights text-relevant regions
- Original image features are adaptively weighted by attention

---

### 5. **Decoder: Atrous Separable Convolution**

**Purpose**: Refine fused features and output segmentation logits

**Architecture**:
```
Input: Fused Features (B × 512 × 64 × 64)

Step 1: Atrous Separable Convolution
    ├─ Depthwise Convolution
    │  ├─ Kernel: 3×3, Dilation: 2, Padding: 2
    │  ├─ Groups: in_channels (groups = 512)
    │  └─ Output: (B × 512 × 64 × 64)
    ├─ Pointwise Convolution
    │  ├─ 1×1 Conv: 512 → 256
    │  └─ Output: (B × 256 × 64 × 64)

Step 2: Batch Normalization
    └─ Output: (B × 256 × 64 × 64)

Step 3: GELU Activation
    └─ Output: (B × 256 × 64 × 64)

Output: Refined Features (B × 256 × 64 × 64)
```

**Multi-Scale Receptive Field**:
- Dilated convolution captures features at multiple scales
- Effective receptive field without pooling (preserves spatial resolution)
- Suitable for precise medical image segmentation

---

## Detailed Encoding Pipeline

### Stage 1: Visual Encoding

```python
# Input: Medical image (B × 3 × 256 × 256)
image → RepViT encoder
  ├─ Patch embedding: 256 → 16 patches per side (4×4 per patch)
  ├─ Patch features: 16 × 16 = 256 spatial locations
  ├─ 8 Vision Transformer blocks
  │  └─ Each: Self-attention + MLP with residuals
  └─ Output: (B × 256 × 64 × 64) feature map
```

### Stage 2: Text Encoding

```python
# Input: Text prompt (e.g., "left kidney")
text → Tokenization (max 256 tokens)
  → PubMedBERT (12 layers)
    ├─ Embedding layer: token → 768-dim
    ├─ 12 transformer blocks
    │  └─ Multi-head self-attention (12 heads)
    ├─ [CLS] token extraction
    └─ Linear projection: 768 → 1024
  → Output: (B × 1 × 1024) text embedding
```

---

## Detailed Decoding Pipeline

### Stage 1: ShuffleFormer Fusion

```python
# Input: Visual (B × 64 × 64 × 256), Text (B × 1 × 1024)
x = rearrange(visual, "b c h w → b (h w) c")  # Flatten to (B × 4096 × 256)

# Token Shuffling
x = shuffle(x)  # (B × 1024 × 256) - reduced spatial, grouped tokens

# Multi-head Transformer with Text Guidance
for layer in transformer_blocks:
    x = concat(text, x)  # (B × 1025 × 256)
    x = LayerNorm(x)
    x = x + MultiHeadAttention(x)  # Text-guided attention
    text_token = x[0]  # Extract updated text
    x = x[1:]  # Remove text
    text = α * text + (1 - α) * text_token  # EMA update
    x = LayerNorm(x)
    x = x + MLP(x)

# Channel Unshuffling
x = unshuffle(x)  # (B × 4096 × 64) → (B × 64 × 64 × 64)
x = rearrange(x, "b (h w) c → b c h w")
```

### Stage 2: Attention Gating

```python
# Input: Image features x (B × 448 × 64 × 64), Text-guided g (B × 64 × 64 × 64)
x_proj = Conv2d(448 → 256)(x)
g_proj = Conv2d(64 → 256)(g)
attn = Sigmoid(Conv2d(256 → 1)(GELU(x_proj + g_proj)))  # (B × 1 × 64 × 64)
output = concat(attn * x, g)  # (B × 512 × 64 × 64)
```

### Stage 3: Decoder Convolution

```python
# Input: Fused features (B × 512 × 64 × 64)
output = BatchNorm(Conv2d_Atrous(512 → 256, kernel=3, dilation=2))
output = GELU(output)  # (B × 256 × 64 × 64)

# Segmentation head (not shown here)
logits = Conv2d(256 → num_classes)(output)  # (B × num_classes × 256 × 256)
```

---

## Training Configuration

### Default Configuration (`configs/text_seg_repvit.yaml`)

```yaml
model:
  target: training.train_textsam.TextSAM
  base_learning_rate: 4.5e-5
  
  image_encoder:
    target: src.models.visual.image_encoder.repvit_m1_0
  
  text_embedder_configs:
    version: ckpts/bert
    max_length: 256
    d_model: 768
    output_dim: 1024
    local_pt: ckpts/pubmedbert.pt
  
  maskformer_config:
    img_size: 64
    patch_size: 4
    embed_dim: 256
    depth: 2
    num_heads: 8
    mlp_ratio: 0.5
    n_mlp_blocks: 2

data:
  batch_size: 5
  num_workers: 8
  train:
    data_dir: /path/to/train_npy
    text_label_path: /path/to/text_seg_class.json
    image_size: 256
    n_slicing: 9

lightning:
  trainer:
    max_epochs: 100
    accelerator: gpu
```

---

## Usage Guide

### 1. Data Preparation

**Convert NPZ to NPY:**
```bash
python utils/npz_to_npy.py \
    -npz_dir data/npz/CT_Abd \
    -npy_dir data/npy \
    -num_workers 4
```

**Expected data structure:**
```
data/npy/
├── imgs/          # Normalized images [0, 1]
│   ├── case_001-000.npy
│   └── ...
├── gts/           # Ground truth masks
│   ├── case_001-000.npy
│   └── ...
└── embeddings/    # (Optional) Pre-computed embeddings
    └── ...
```

### 2. Training

**Default configuration:**
```bash
python main.py \
    -t True \
    -b configs/text_seg_repvit.yaml \
    -p Text-MedSAM \
    -s 42
```

**Single GPU training:**
```bash
python train_one_gpu.py \
    -data_root /path/to/train_npy \
    -pretrained_checkpoint ckpts/rep_medsam.pth \
    -num_epochs 100 \
    -batch_size 16 \
    -lr 4.5e-5
```

**With knowledge distillation:**
```bash
bash scripts/embeddings.sh

python train_one_gpu.py \
    -data_root /path/to/train_npy \
    -embedding_path /path/to/embeddings \
    -distillation True
```

### 3. Inference

**Batch inference:**
```bash
python scripts/batch_infer_text.py \
    --checkpoint_path ckpts/model.ckpt \
    --img_dir /path/to/images \
    --gt_dir /path/to/ground_truth \
    --output_dir /path/to/outputs
```

### 4. Multi-GPU Training (DDP)

```bash
python main.py \
    -t True \
    -b configs/text_seg_repvit.yaml \
    -p Text-MedSAM \
    -s 42
```

---

## Key Technical Innovations

### 1. **Efficient Text-Image Fusion**
- Token shuffling reduces computational overhead by grouping spatially-proximal features
- Channel unshuffling preserves spatial structure for precise segmentation
- Adaptive fusion via EMA-updated text embeddings

### 2. **Cross-Modal Attention**
- Text embedding guides visual attention without increasing parameters
- Learnable fusion parameters for task-specific adaptation
- Bidirectional information flow between modalities

### 3. **Lightweight Architecture**
- RepViT: 2.7× faster than TinyViT with comparable parameters
- Atrous separable convolutions: Multi-scale features without pooling
- Efficient memory utilization for real-time inference

### 4. **Medical Domain Optimization**
- PubMedBERT: Pre-trained on 4.5B+ biomedical tokens
- Anatomical text prompts for precise region specification
- Knowledge distillation from teacher models for improved accuracy

---

## Performance Benchmarks

### Efficiency on 3D Volumes

| Case | Volume Size | Inference Time |
|------|------------|-----------------|
| CT_0566 | 287×512×512 | 73.64s |
| CT_0888 | 237×512×512 | 25.09s |
| MR_0121 | 64×290×320 | 16.16s |

**~2-3× speedup** achieved with optimized inference pipeline

---

## Citation

If you use Text-MedSAM in your research, please cite:

```bibtex
@misc{textmedsam,
  title={Text-MedSAM: Medical Image Segmentation Guided by Text Prompts},
  author={Wei, Muxin},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/muxin-wei/Text-MedSAM}}
}
```

---

## License

[Add your license information here]

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

---

## Acknowledgments

- RepViT architecture from [ADD SOURCE]
- PubMedBERT from Microsoft Research
- MedSAM framework inspiration
