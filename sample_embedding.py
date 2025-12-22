import os
import os.path as osp
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# --- IMPORT SETUP ---
# Manually add project root to sys.path to resolve 'src' and 'dataset' imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming these imports are now resolved:
from src.repvit import repvit_m1_0 
from dataset.text_seg_n import TextSeg # Assuming TextSeg is the original class
from dataset.utils import process_input # Assuming this function is available

# --- DDP Setup and Cleanup ---

def setup(rank, world_size):
    """Initialize DDP environment."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy DDP process group."""
    dist.destroy_process_group()


def custom_embedding_collate(batch_list):
    """
    Custom collate function to handle variable slice count (D) per volume.
    Flattens all slices from all volumes into a single tensor for the model.
    """
    batch_list = [b for b in batch_list if b is not None]
    
    all_names = []
    all_slice_counts = []
    all_flat_images = []
    
    for item in batch_list:
        # item['image'] shape: (D, H, W)
        D, H, W = item['image'].shape
        
        # Add channel dim: (D, 1, H, W)
        all_flat_images.append(item['image'].unsqueeze(1)) 

        all_names.append(item['img_name'])
        all_slice_counts.append(D)
        
    total_flat_images = torch.cat(all_flat_images, dim=0)

    total_flat_images = total_flat_images.repeat(1, 3, 1, 1)
    
    return {
        "images": total_flat_images,
        "slice_counts": all_slice_counts,
        "img_names": all_names
    }

# --- Main DDP Embedding Function ---

def compute_embeddings_ddp(rank, world_size, args):
    """Main function to run DDP embedding computation."""
    
    # 1. SETUP DDP
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier() 
        
    # 2. DATASET and DATALOADER
    # Assuming TextSeg is initialized to return D slices, not N fixed slices.
    dataset = TextSeg(
        data_dir=args.data_dir,
        text_label_path=args.label_path,
        image_size=args.image_size
        # Note: We must ensure TextSeg.__getitem__ returns all D slices (D, H, W) 
        # for a volume, and not a fixed number (N).
    )
    
    # Use DistributedSampler for DDP
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_embedding_collate # Use the custom collate function
    )
    model = repvit_m1_0()

    CHECKPOINT_PATH = "/root/autodl-tmp/Rep-MedSAM/ckpts/rep_medsam.pth"
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if world_size > 0 else 'cpu'
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace('image_encoder.', ''): v for k, v in state_dict.items() if k.startswith('image_encoder')}
    model.to(rank) 
    model.load_state_dict(state_dict, strict=True)
    
    if rank == 0:
        print(f"Rank 0: Successfully loaded checkpoint from {CHECKPOINT_PATH}")
    
    # 3. MODEL SETUP
    # DDP wrapping is now safe since model is on GPU:rank
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.eval()
    
    # 4. EMBEDDING COMPUTATION (Inference)
    all_embeddings = []
    all_names = []
    all_slice_counts = []
    
    if rank == 0:
        print(f"Starting embedding computation on {world_size} processes...")

    # Disable gradient computation
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank} Processing")):
            if batch is None:
                continue

            images = batch['images'].to(device) 
            embeddings = ddp_model(images)  
            embeddings = embeddings.cpu().numpy().astype(np.float32)
            
            slice_counts_np = np.array(batch['slice_counts'])
            split_indices = np.cumsum(slice_counts_np)
            relative_path = osp.relpath(file_path, args.data_dir)
            output_file_path = osp.join(args.output_dir, relative_path)
            split_embeddings = np.split(embeddings, split_indices[:-1])
            for vol_name, vol_embeddings in zip(batch['img_names'], split_embeddings[:-1]):
                output_file = osp.join(args.output_dir, f"{vol_name}.npz") 
                np.save(
                    output_file, 
                    vol_embeddings,
                )

    cleanup()

# --- Arg Parsing and Entry Point ---

def main_cli():
    parser = argparse.ArgumentParser(description="DDP Image Embedding Computation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the text label JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output embeddings.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (number of volumes) per GPU.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    
    args = parser.parse_args()
    
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Error: Script must be launched with torchrun.")
        sys.exit(1)
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Wait for Rank 0 to create the directory    
    compute_embeddings_ddp(rank, world_size, args)


if __name__ == "__main__":
    main_cli()