# Text-MedSAM: Text-Guided Medical Image Segmentation

A text-guided medical image segmentation framework built on the MedSAM architecture. This repository implements medical image segmentation using natural language prompts combined with efficient vision transformers.

## Overview

Text-MedSAM extends the Segment Anything Model (SAM) for medical imaging by incorporating text prompts as guidance for segmentation tasks. The framework leverages:

- **Efficient Image Encoding**: RepViT-based encoder for fast processing
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

3. **Docker Setup** (Optional)
   ```bash
   docker build -t text-medsam .
   docker run -it --gpus all -v /path/to/data:/inputs -v /path/to/outputs:/outputs text-medsam
   ```

## Repository Structure

```
Text-MedSAM/
├── src/                           # Core source code
│   ├── dataset/
│   │   ├── textseg.py            # Text-guided segmentation dataset loader
│   │   └── utils.py              # Data processing utilities
│   ├── repvit.py                 # RepViT model architecture
│   └── models/                   # SAM components and custom layers
├── scripts/                       # Training and inference scripts
│   ├── embeddings.sh             # Distributed embedding generation (DDP)
│   ├── text_predict.sh           # Batch inference script
│   └── batch_infer_text.py       # Inference implementation
├── utils/
│   ├── utils.py                  # General utilities (MetricLogger, SmoothedValue)
│   └── npz_to_npy.py            # NPZ → NPY format conversion
├── ckpts/                         # Model checkpoints directory
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

- **`src/repvit.py`**: RepViT model implementation
  - Efficient lightweight image encoder
  - 2.7× faster than TinyViT with comparable parameters
  - Optimized for medical image analysis

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
- Pre-trained model weights
- Fine-tuned checkpoints
- Teacher model weights (for distillation)

## Usage Guide

### 1. Data Preparation

**Convert NPZ to NPY format:**
```bash
python utils/npz_to_npy.py \
    -npz_dir data/npz/MedSAM_train/CT_Abd \
    -npy_dir data/npy \
    -num_workers 4
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

**Standard single-GPU training:**
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
    -b config.yaml \
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

### Prompt Encoder & Mask Decoder
- Adapted from SAM for medical imaging
- Support for text prompts (in addition to bounding boxes/points)
- Multi-task learning with classification head

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

## Configuration Files

The project uses YAML configuration files (referenced in `main.py`). See `-b config.yaml` parameter for custom configuration paths.

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

### Data Loading Issues
- Ensure NPY files are properly generated from NPZ
- Verify text label JSON format matches expected structure
- Check file path consistency between images and ground truth

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
- All contributors and users providing feedback and improvements
