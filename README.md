# Text-MedSAM: Text-Guided Medical Image Segmentation

A text-guided medical image segmentation framework built on the MedSAM architecture. This repository implements medical image segmentation using natural language prompts combined with efficient vision transformers.

## Overview

Text-MedSAM extends the Segment Anything Model (SAM) for medical imaging by incorporating text prompts as guidance for segmentation tasks. The framework leverages:

- **Efficient Image Encoding**: RepViT-based encoder for fast processing
- **Medical Text Encoding**: PubMedBERT for biomedical text understanding
- **Text-Guided Prompting**: Natural language descriptions for anatomical structures
- **Multi-Modal Learning**: Combines visual and textual information for improved segmentation
- **Distributed Training**: DDP support for efficient large-scale model training

### Key Features

- ✅ Text-guided segmentation of medical images (CT, MRI, PET, etc.)
- ✅ Support for both 2D and 3D medical imaging data
- ✅ Distributed Data Parallel (DDP) training on multi-GPU systems
- ✅ Knowledge distillation from teacher models
- ✅ Efficient inference with optimized model architecture
- ✅ Comprehensive data preprocessing utilities
- ✅ Medical-domain text embeddings with PubMedBERT

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
   # Create checkpoint directory
   mkdir -p ckpts
   
   # Download PubMedBERT (place in ckpts/)
   # wget -O ckpts/pubmedbert.pt <model_url>
   ```

4. **Docker Setup** (Optional)
   ```bash
   docker build -t text-medsam .
   docker run -it --gpus all -v /path/to/data:/inputs -v /path/to/outputs:/outputs text-medsam
   ```

## Repository Structure

```
Text-MedSAM/
├── configs/                       # Configuration files
│   └── text_seg_repvit.yaml      # Default training configuration
├── src/                           # Core source code
│   ├── dataset/
│   │   ├── textseg.py            # Text-guided segmentation dataset loader
│   │   └── utils.py              # Data processing utilities
│   ├── models/
│   │   ├── visual/
│   │   │   └── image_encoder.py  # RepViT encoder
│   │   ├── heads/
│   │   │   └── text_embedder.py  # Text encoder (PubMedBERT)
│   │   └── maskformer.py         # MaskFormer decoder
│   ├── losses/
│   │   ├── total_loss.py         # Combined loss function
│   │   ├── medsam_loss.py        # Segmentation loss
│   │   └── clip_loss.py          # CLIP-based alignment loss
│   └── repvit.py                 # RepViT model architecture
├── scripts/                       # Training and inference scripts
│   ├── embeddings.sh             # Distributed embedding generation (DDP)
│   ├── text_predict.sh           # Batch inference script
│   └── batch_infer_text.py       # Inference implementation
├── utils/
│   ├── utils.py                  # General utilities (MetricLogger, SmoothedValue)
│   └── npz_to_npy.py            # NPZ → NPY format conversion
├── ckpts/                         # Model checkpoints directory
│   ├── rep_medsam.pth            # Pre-trained image encoder
│   ├── pubmedbert.pt             # Text encoder checkpoint
│   └── bert/                     # BERT vocabulary
├── main.py                        # Lightning-based training framework
├── train_one_gpu.py              # Single GPU training script
├── sample_embedding.py            # DDP embedding computation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── Dockerfile                    # Docker container configuration
└── README.md                     # This file
```

## Folder Descriptions

### `src/` - Core Implementation

- **`src/dataset/`**
  - `textseg.py`: Main dataset class for text-guided segmentation
    - Handles NPZ file loading with text annotations
    - Supports 2D and 3D medical image volumes
    - Applies augmentations during training
    - Returns: image tensor, segmentation mask, text prompts, class IDs

- **`src/dataset/utils.py`**: Helper functions for:
  - Image resizing and padding
  - Random transformations and augmentations
  - Data preprocessing pipeline

- **`src/models/visual/image_encoder.py`**: RepViT implementation
  - Efficient lightweight image encoder
  - 2.7× faster than TinyViT with comparable parameters
  - Optimized for medical image analysis
  - Input: 256×256 images, Output: 64×64 feature maps

- **`src/models/heads/text_embedder.py`**: Text encoder component
  - **PubMedBERT**: Specialized BERT model trained on biomedical literature
  - Converts text prompts to high-dimensional embeddings
  - See [Text Encoder Details](#text-encoder-architecture) for specifications

- **`src/models/maskformer.py`**: MaskFormer decoder
  - Decodes combined visual and textual features
  - Generates segmentation masks
  - Supports multi-class and instance segmentation

- **`src/losses/`**: Loss functions
  - `medsam_loss.py`: Primary segmentation loss (BCE + Dice)
  - `clip_loss.py`: Text-image alignment loss
  - `total_loss.py`: Combined loss function with weighted components

### `scripts/` - Training & Inference

- **`embeddings.sh`**: Multi-GPU embedding generation
  - Uses `torchrun` for DDP execution
  - Generates image embeddings from MedSAM encoder
  - Configurable batch size and number of workers
  - Required for knowledge distillation training

- **`text_predict.sh`**: Batch inference script
  - Iterates over all checkpoints in directory
  - Performs inference on validation data
  - Outputs segmentation predictions

- **`batch_infer_text.py`**: Inference implementation
  - Loads model from checkpoint
  - Processes images with text prompts
  - Generates and saves predictions

### `configs/` - Configuration Files

- **`text_seg_repvit.yaml`**: Default training configuration
  - Specifies model architecture and parameters
  - Defines data loading configuration
  - Sets loss weights and training hyperparameters
  - See [Default Configuration](#default-configuration) for details

### `utils/` - Utilities

- **`utils.py`**: Training utilities
  - `SmoothedValue`: Running average tracking with median/mean
  - `MetricLogger`: Aggregates and synchronizes metrics across GPUs
  - DDP synchronization helpers

- **`npz_to_npy.py`**: Data format conversion
  - Converts preprocessed NPZ files to NPY format
  - Handles both 2D and 3D medical images
  - Supports parallel conversion with multiprocessing
  - Normalizes images to [0, 1] range
  - Usage: `python npz_to_npy.py -npz_dir <input> -npy_dir <output>`

### `ckpts/` - Model Checkpoints

Directory for storing:
- `rep_medsam.pth`: Pre-trained RepViT image encoder
- `pubmedbert.pt`: PubMedBERT text encoder weights
- `bert/`: BERT tokenizer vocabulary files
- Fine-tuned checkpoints from training runs

## Text Encoder Architecture

### Overview
The Text Encoder is responsible for converting medical text prompts into semantic embeddings that guide the segmentation model.

### PubMedBERT Component

**Purpose**: Extract medical domain knowledge from text descriptions

**Architecture Details**:
- **Model Type**: BERT variant pre-trained on PubMed biomedical literature
- **Vocabulary**: Medical and biomedical terminology
- **Output Dimension**: 768 (hidden states)
- **Configuration**:
  - `version`: BERT model variant (ckpts/bert)
  - `max_length`: 256 tokens maximum
  - `d_model`: 768 hidden dimensions
  - `output_dim`: 1024 final embedding dimension (after projection)
  - `local_pt`: Checkpoint path (ckpts/pubmedbert.pt)

**Key Advantages**:
- Trained on 4.5B+ tokens from PubMed abstracts
- Better understanding of medical terminology
- Captures anatomical and pathological context
- Improved over general-purpose BERT for medical tasks

**Processing Pipeline**:
1. Tokenize text prompt to tokens ≤ 256 length
2. Pass through PubMedBERT (d_model=768)
3. Apply projection layer to output_dim=1024
4. Project to match visual feature dimensions for fusion

### Text Prompt Format

Medical anatomical structures are described in natural language:
```
Examples of valid text prompts:
- "left kidney"
- "liver with focal lesion"
- "segmentation of cardiac ventricle"
- "pancreatic head region"
```

Text labels are managed in a JSON configuration file:
```json
{
  "CT_Abd": {
    "1": ["kidney", "left kidney", "renal cortex"],
    "2": ["liver", "hepatic parenchyma", "liver tissue"],
    "instance_label": 0
  }
}
```

## Default Configuration

The default training configuration is defined in `configs/text_seg_repvit.yaml`. Here are the key settings:

### Model Configuration

```yaml
model:
  target: training.train_textsam.TextSAM
  base_learning_rate: 4.5e-5
  
  # Image Encoder (RepViT)
  image_encoder:
    target: src.models.visual.image_encoder.repvit_m1_0
  
  # Text Encoder (PubMedBERT)
  text_embedder_configs:
    target: src.models.heads.text_embedder.TextEmbedder
    params:
      version: ckpts/bert
      max_length: 256              # Maximum text token length
      d_model: 768                 # PubMedBERT hidden dimension
      output_dim: 1024             # Final embedding dimension
      local_pt: ckpts/pubmedbert.pt  # Pretrained checkpoint
  
  # MaskFormer Decoder
  maskformer_config:
    target: src.models.maskformer.MaskFormer
    params:
      img_size: 64                 # Feature map size (256/4)
      patch_size: 4               # Patch size for tokenization
      in_chans: 256               # Input channels from encoder
      embed_dim: 1024             # Embedding dimension
      depth: 4                    # Number of transformer layers
      num_heads: 8                # Number of attention heads
  
  # Loss Configuration
  loss_configs:
    target: src.losses.total_loss.TextSegLoss
    params:
      clip_loss_weight: 1.0       # Text-image alignment loss weight
      bg_loss_weight: 1.0         # Background loss weight
      seg_loss_configs:
        target: src.losses.medsam_loss.MedSamLoss
        params:
          reduction: "mean"       # Loss reduction method
      clip_loss_configs:
        target: src.losses.clip_loss.CLIPLoss
        params:
          temperature: 0.07       # Temperature for contrastive loss
  
  # General Parameters
  image_size: 256                 # Input image resolution
  hidden_dim: 256                 # Hidden dimension for processing
  checkpoint: /root/autodl-tmp/Rep-MedSAM/ckpts/rep_medsam.pth
```

### Data Configuration

```yaml
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 5                 # Batch size per GPU
    num_workers: 8                # DataLoader workers
    train:
      target: dataset.textseg.TextSeg
      params:
        data_dir: /root/autodl-tmp/dataset/train  # Training data path
        text_label_path: /root/autodl-tmp/dataset/text_seg_class.json
        image_size: 256
        n_slicing: 9              # Number of slices to sample per 3D volume
```

### Training Configuration

```yaml
lightning:
  callbacks:
    image_logger:
      target: utils.helper.ImageLogger
      params:
        batch_frequency: 200      # Log every N batches
        max_images: 4             # Max images per log
        increase_log_steps: False
  
  trainer:
    max_epochs: 100               # Maximum training epochs
    accelerator: gpu              # Use GPU acceleration
    num_sanity_val_steps: 0       # Skip sanity check
    limit_val_batches: 0          # Disable validation during training
    num_nodes: 1                  # Single node training
```

### Using Custom Configuration

To train with the default configuration:

```bash
python main.py \
    -t True \
    -b configs/text_seg_repvit.yaml \
    -p Text-MedSAM \
    -s 42
```

To create a custom configuration, copy and modify `text_seg_repvit.yaml`:

```bash
cp configs/text_seg_repvit.yaml configs/text_seg_custom.yaml
# Edit paths and parameters as needed
python main.py -b configs/text_seg_custom.yaml
```

## Usage Guide

### 1. Data Preparation

**Convert NPZ to NPY format:**
```bash
python utils/npz_to_npy.py \
    -npz_dir data/npz/MedSAM_train/CT_Abd \
    -npy_dir data/npy \
    -num_workers 4
```

**Prepare text labels JSON file:**
```json
{
  "dataset_name": {
    "class_id": ["label1", "label2", "label3"],
    "instance_label": 0
  }
}
```

**Expected data structure:**
```
data/npy/
├── imgs/          # Training images (normalized to [0, 1])
│   ├── case_001-000.npy
│   ├── case_001-001.npy
│   └── ...
├── gts/           # Ground truth segmentation masks
│   ├── case_001-000.npy
│   ├── case_001-001.npy
│   └── ...
└── embeddings/    # (Optional) Pre-computed image embeddings
    ├── case_001-000.npy
    └── ...
```

### 2. Training

**Using default configuration:**
```bash
python main.py \
    -t True \
    -b configs/text_seg_repvit.yaml \
    -p Text-MedSAM \
    -s 42
```

**Single GPU training (legacy):**
```bash
python train_one_gpu.py \
    -data_root /path/to/train_npy \
    -pretrained_checkpoint ckpts/rep_medsam.pth \
    -work_dir work_dir \
    -num_epochs 10 \
    -batch_size 16 \
    -num_workers 12
```

**With knowledge distillation:**

First, generate embeddings from teacher model:
```bash
bash scripts/embeddings.sh
```

Then train with distillation:
```bash
python train_one_gpu.py \
    -data_root /path/to/train_npy \
    -embedding_path /path/to/embeddings \
    -mask_decoder teacher_model/mask_decoder.pth \
    -prompt_encoder teacher_model/prompt_encoder.pth \
    -work_dir work_dir/distillation \
    -num_epochs 5 \
    -lr 5E-4 \
    -distillation True \
    -num_workers 12
```

### 3. Inference

**Batch inference on validation set:**
```bash
bash scripts/text_predict.sh
```

Or run directly:
```bash
python scripts/batch_infer_text.py \
    --checkpoint_path ckpts/model.ckpt \
    --img_dir /path/to/images \
    --gt_dir /path/to/ground_truth \
    --output_dir /path/to/outputs \
    --image-size 256 \
    --ds-scale 4.0
```

### 4. Distributed Training (Multi-GPU)

Using PyTorch Lightning with DDP:
```bash
python main.py \
    -t True \
    -b configs/text_seg_repvit.yaml \
    -p Text-MedSAM \
    -s 42
```

### 5. Embedding Generation (DDP)

Generate embeddings for distillation on multiple GPUs:
```bash
bash scripts/embeddings.sh
# Configure NUM_GPUS, DATA_DIR, and OUTPUT_DIR in the script
```

## Training Parameters

Key command-line arguments for `train_one_gpu.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-data_root` | str | `/mnt/train` | Path to NPY training data |
| `-pretrained_checkpoint` | str | `workdir/finetune/finetune_weights.pth` | Pre-trained model weights |
| `-work_dir` | str | `./workdir/finetune/` | Directory for checkpoints and logs |
| `-num_epochs` | int | 20 | Number of training epochs |
| `-batch_size` | int | 16 | Batch size per GPU |
| `-num_workers` | int | 12 | DataLoader workers |
| `-lr` | float | 1E-4 | Learning rate |
| `-bbox_shift` | int | 5 | Bounding box perturbation for augmentation |
| `-distillation` | bool | False | Enable knowledge distillation |
| `-embedding_path` | str | None | Path to pre-computed embeddings |
| `-use_wandb` | bool | False | Log with Weights & Biases |

## Model Architecture

### Image Encoder
- **RepViT**: Lightweight and efficient vision transformer
- Parameters: ~6M (vs. 5.7M for TinyViT)
- Inference Latency: 0.36s (vs. 0.98s for TinyViT)
- Output: 64×64×256 feature maps

### Text Encoder
- **PubMedBERT**: Biomedical domain-specific BERT
- Architecture: 12 layers, 768 hidden dimension
- Pre-trained on 4.5B+ PubMed tokens
- Output: 1024-dimensional embeddings

### Decoder (MaskFormer)
- **Input**: Fused visual and textual features
- **Architecture**: 4-layer transformer
- **Output**: Per-pixel classification logits
- **Support**: Multi-class and instance segmentation

### Knowledge Distillation
- Teacher: Original MedSAM model
- Student: Lightweight Rep-MedSAM
- Distillation Loss: MSE between embeddings

## Performance Metrics

Efficiency improvements on 3D volumes:

| Case | Volume Size | Baseline (s) | Rep-MedSAM (s) | Updated (s) |
|------|------------|--------------|----------------|-------------|
| CT_0566 | 287×512×512 | 436.97 | 194.22 | **73.64** |
| CT_0888 | 237×512×512 | 115.53 | 53.44 | **25.09** |
| MR_0121 | 64×290×320 | 119.09 | 54.62 | **16.16** |

**~2-3× speedup achieved with updated inference pipeline**

## Troubleshooting

### CUDA Memory Issues
```bash
# Reduce batch size
python train_one_gpu.py -batch_size 8 ...

# Enable gradient accumulation in training script
```

### DDP Issues
```bash
# Set PYTORCH allocation strategy
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Text Encoder Issues
- Ensure PubMedBERT checkpoint exists at `ckpts/pubmedbert.pt`
- Verify BERT vocabulary files in `ckpts/bert/`
- Check text label JSON format is valid

### Data Loading Issues
- Ensure NPY files are properly generated from NPZ
- Verify text label JSON format matches expected structure
- Check file path consistency between images and ground truth
- Confirm text prompts are in the label JSON file

## Citation

If you use this project in your research, please cite:

```bibtex
@article{wei2024textmedsam,
  title={Text-MedSAM: Text-Guided Medical Image Segmentation},
  author={Wei, Muxin and Chen, Shuqing and Wu, Silin and Xu, Dabin},
  year={2024}
}
```

Based on:
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [RepViT](https://github.com/THU-MIG/RepViT)
- [PubMedBERT](https://github.com/microsoft/BiomedNLP-PubMedBERT)

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and questions:
- Open a GitHub issue
- Check existing issues for solutions
- Review the troubleshooting section above

## Acknowledgments

- **MedSAM Team** (bowang-lab) for the foundational segmentation model
- **RepViT Team** (THU-MIG) for the efficient vision transformer
- **Microsoft Research** for PubMedBERT biomedical language model
- All contributors and users providing feedback and improvements
