#!/bin/bash
# run_2gpu_embedding.sh

# -----------------------------------------------------------
# 1. ENVIRONMENT SETUP
# -----------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
NEW_PORT=30005
# Define the number of GPUs to use (Set to 2 as requested)
NUM_GPUS=2

# Define the path to the Python script
PYTHON_SCRIPT="sample_embedding.py"

DATA_DIR="/root/autodl-tmp/dataset/train" 

# Path to the text label JSON file
LABEL_PATH="/root/autodl-tmp/dataset/text_seg_class.json"

OUTPUT_DIR="/root/autodl-tmp/dataset/train_embeddings" 

# Execution parameters
BATCH_SIZE=2      # Number of volumes per GPU in a batch
IMAGE_SIZE=256      # Target image size
NUM_WORKERS=4       # DataLoader workers per GPU

echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "Starting DDP embedding computation on ${NUM_GPUS} GPUs..."

# Execute the Python script using torchrun
# --nproc_per_node: Specifies the number of processes (GPUs) to launch
torchrun --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${NEW_PORT} \
    "${PYTHON_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --label_path "${LABEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --num_workers "${NUM_WORKERS}"

echo "DDP Script finished successfully. Results saved in ${OUTPUT_DIR}"