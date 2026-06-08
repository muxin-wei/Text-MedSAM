# Text-MedSAM: ShuffleFormer Branch

## Overview

The **shuffleformer** branch represents a specialized implementation of the Text-MedSAM project with a focus on the **ShuffleFormer** architecture for efficient medical image segmentation with text guidance.

### Repository Details
- **Project**: Text-MedSAM
- **Owner**: muxin-wei
- **License**: Apache License 2.0
- **Language**: Python (99.1%)
- **Default Branch**: master
- **Specialized Branch**: shuffleformer

---

## Branch Architecture

### Key Differences from Main Branch

This branch introduces significant architectural changes:

| Aspect | shuffleformer | master |
|--------|---------------|--------|
| **Primary Model** | ShuffleFormer | MaskFormer, SAM |
| **Model Files** | `shuffle_former.py` (NEW) | `maskformer.py`, `sam.py` |
| **Optimization Focus** | Token shuffling & channel unshuffling | Standard transformer architectures |
| **Approach** | Lightweight, efficient attention mechanism | Full-resolution attention |

---

## Core Components

### 1. ShuffleFormer Architecture (`src/models/shuffle_former.py`)

The main model implementing the ShuffleFormer architecture with the following key classes:

#### **ShuffleFormer** (Main Model)
```
Input: 
  - img_size: int = 64
  - patch_size: int = 4
  - embed_dim: int = 256
  - depth: int = 2
  - num_heads: int = 8
  - in_chans: List[int] = [448, 112, 56] (multi-scale inputs)

Output:
  - x: fused features
  - text: updated text embeddings
  - attn_weights: attention weights from all layers
  - out_x: intermediate outputs from each layer
```

**Key Features:**
- Supports 3 parallel attention layers for multi-scale feature processing
- Absolute positional embeddings (optional)
- Text-Image EMA (Exponential Moving Average) fusion mechanism
- Efficient token shuffling and channel operations

#### **ShuffleAttention** (Core Attention Block)
Multi-scale attention mechanism combining:
- **TokenShuffle**: Reduces computational complexity through token dimensionality reduction
- **TransformerBlock**: Self-attention with RMSNorm and MLP layers
- **ChannelUnshuffle**: Reverses shuffling to restore spatial dimensions
- **AttenGate**: Gating mechanism for feature refinement
- **AtrousSeparableConvolution**: Dilated convolutions for multi-scale feature capture

#### **TokenShuffle**
```python
def forward(x):
    # Input: B, N, C (batch, num_tokens, channels)
    # MLP projects: C -> C // s^2 (where s is shuffle size)
    # Rearrange tokens into blocks
    # Optional fusion with MLPs
    # Output: B, N // s^2, C (reduced token count)
```

**Purpose**: Dramatically reduces computational cost by grouping and compressing tokens

#### **ChannelUnshuffle**
Inverse operation of TokenShuffle:
- Restores spatial dimensions
- Projects back to original channel dimensions
- Supports optional channel-wise fusion

#### **Supporting Modules**

**TransformerBlock**
- Self-attention layer (from `transformer.py`)
- MLP with GELU activation
- RMSNorm for pre-norm architecture
- Residual connections

**AttenGate** (Attention Gating)
- Computes attention weights between feature map `x` and gated signal `g`
- Applies sigmoid gating: `attn = sigmoid(ψ(gelu(x + g)))`
- Multiplicative attention: `output = attn_weight × x_org`

**AtrousSeparableConvolution**
- Separable atrous (dilated) convolution
- Efficient convolution with dilation rate
- Reduces computation while maintaining receptive field

**Text2Mask**
- Converts image-text similarity to segmentation masks
- Normalized dot-product similarity
- Spatial rearrangement to mask dimensions

### 2. Supporting Modules (`src/models/modules.py`)

Common utility modules shared across models:

#### **MLPBlock**
Standard MLP: Linear → GELU → Linear

#### **PatchEmbed**
Image to Patch Embedding using convolution:
- Converts 2D images to patch embeddings
- Configurable kernel size, stride, padding

#### **LayerNorm2d**
2D LayerNorm (used as pre-norm in transformer blocks)

#### **Conv2d_BN**
Fused Conv2d + BatchNorm2d with weight initialization

#### **Residual**
Residual wrapper with optional stochastic depth

#### **RepVGGDW** & **RepViTBlock**
Advanced building blocks for efficient vision models

#### **BN_Linear**
Fused BatchNorm1d + Linear layer

### 3. Transformer Utilities (`src/models/transformer.py`)

Provides base attention mechanisms used by ShuffleFormer.

---

## Key Design Innovations

### 1. **Efficient Token Shuffle**
```
Original tokens: [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈]
After MLP reduction: [t'₁, t'₂] (compressed to 1/4)
After spatial rearrangement: organized blocks
Final: Fewer tokens for attention computation
```

**Benefit**: Reduces attention complexity from O(N²) to approximately O((N/s²)²)

### 2. **Multi-Scale Processing**
```
Three parallel paths with different shuffle sizes:
- Path 1: in_chans=448, shuffle_size=2, unshuffle_size=8
- Path 2: in_chans=112, shuffle_size=4, unshuffle_size=16  
- Path 3: in_chans=56,  shuffle_size=4, unshuffle_size=1

Each path processes different feature scales independently
```

### 3. **Text-Image Fusion with EMA**
```
text_new = α × text_old + (1-α) × text_from_attention
where α = exp(text_ema).clamp(0.0, 0.999)
```

Learnable parameter `text_ema` controls the balance between:
- Preserving original text embeddings
- Updating with image-guided information

### 4. **Attention Gating Mechanism**
Combines spatial features with attention gates:
- Projects features to intermediate dimension
- Computes attention map via sigmoid gating
- Applies multiplicative attention for fine-grained control

---

## Forward Pass Flow

```
Input: x (B, C, H, W), text_embed (B, 1, C), feats [f1, f2, f3]

1. Rearrange x to sequence: (B, H×W, C)
2. Add positional embeddings (if use_ape=True)
3. Initialize EMA coefficient α from text_ema

4. For each of 3 layers:
   a. TokenShuffle: reduce token count
   b. Concatenate with text embedding
   c. Apply TransformerBlock (self-attention + MLP)
   d. Extract and update text embedding with EMA
   e. ChannelUnshuffle: restore spatial dimensions
   f. AttenGate: apply attention gating with corresponding feat
   g. AtrousSeparableConv: final refinement
   
5. Return: fused_features, updated_text, attention_weights, all_outputs
```

---

## Configuration Parameters

### ShuffleFormer Initialization
```python
ShuffleFormer(
    img_size=64,              # Input patch size
    patch_size=4,             # Patch division size
    embed_dim=256,            # Embedding dimension
    depth=2,                  # Number of transformer blocks per layer
    num_heads=8,              # Number of attention heads
    mlp_ratio=0.5,            # MLP hidden dim ratio
    use_ape=False,            # Absolute positional embeddings
    attention_downsample_rate=2,
    use_fusion=False,         # Enable channel fusion MLPs
    window_size=[2, 4, 4],    # Shuffle sizes for 3 paths
    uns_size=[8, 16, 1],      # Unshuffle sizes for 3 paths
    n_mlp_blocks=2,           # Number of fusion blocks
    in_chans=[448, 112, 56]   # Input channels for 3 feature scales
)
```

---

## Usage Example

```python
import torch
from src.models.shuffle_former import ShuffleFormer, Text2Mask

# Initialize model
model = ShuffleFormer(
    img_size=64,
    embed_dim=256,
    use_fusion=True
)

# Dummy inputs (batch_size=2)
x = torch.randn(2, 256, 64, 64)           # Image features
text_embed = torch.randn(2, 1, 256)       # Text embeddings (1 token)
feats = [
    torch.randn(2, 448, 64, 64),          # Multi-scale feat 1
    torch.randn(2, 112, 32, 32),          # Multi-scale feat 2
    torch.randn(2, 56, 16, 16)            # Multi-scale feat 3
]

# Forward pass
fused_x, updated_text, attn_weights, intermediate_outs = model(x, text_embed, feats)

# Text-to-mask conversion
text2mask = Text2Mask(init_temp=10.0)
mask_logits = text2mask(fused_x, updated_text)
```

---

## Performance Characteristics

### Computational Efficiency
- **Token Shuffling**: Reduces sequence length from N to N/s² (typically 1/4 to 1/16)
- **Attention Complexity**: O(N²) → O((N/s²)²) per block
- **Memory Usage**: Proportional to reduced token count

### Multi-Scale Processing
- Parallel processing of 3 feature scales
- Independent shuffle parameters for each scale
- Efficient feature fusion via attention gates

### Text-Image Fusion
- Learnable EMA-based fusion
- Preserves text semantic information
- Dynamically adapts to image content

---

## File Structure

```
src/models/
├── shuffle_former.py          # ShuffleFormer implementation (NEW)
├── modules.py                 # Shared utility modules
├── transformer.py             # Attention mechanisms
└── visual/                    # Visual encoding components
```

---

## Advantages Over Standard Transformers

1. **Efficiency**: Significantly fewer tokens through intelligent shuffling
2. **Scalability**: Handles multi-scale features natively
3. **Text Integration**: Specialized fusion mechanism for text-guided segmentation
4. **Memory**: Reduced memory footprint for medical imaging tasks
5. **Accuracy**: Maintains expressiveness through careful architecture design

---

## Related Work

This implementation draws inspiration from:
- Vision Transformers (ViT)
- Efficient attention mechanisms (local attention, token reduction)
- Medical image segmentation with text guidance
- Gating mechanisms for multi-modal fusion

---

## Future Enhancements

- [ ] Add training scripts and loss functions
- [ ] Implement inference pipeline for medical images
- [ ] Add evaluation metrics for segmentation
- [ ] Optimize quantization for deployment
- [ ] Add comprehensive documentation with examples
- [ ] Support for variable image sizes
- [ ] Multi-task learning capabilities

---

## Notes

- **RMSNorm**: Pre-norm architecture uses RMSNorm instead of LayerNorm for improved training stability
- **Einops**: Heavy use of tensor rearrangement operations for efficient reshaping
- **Absolute Positional Embeddings**: Optional feature for incorporating position information
- **Text EMA**: Allows fine-tuning the text-image fusion balance during training

---

**Branch**: shuffleformer  
**Last Updated**: June 8, 2026  
**Language**: Python  
**License**: Apache 2.0
