import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from lightning.fabric import Fabric
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from training.train_text_seg import TextSAM
    from dataset.textseg import ValTextSeg, resize_output
except ImportError as e:
    print(f"Error importing model/dataset classes: {e}")
    sys.exit(1)
    
def restore_prediction_size(pred_seg, gt_shape):
    """
    将预测的分割结果上采样到 GT 的尺寸。
    
    Args:
        pred_seg (torch.Tensor): 预测结果，形状 (D_small, H_small, W_small) 或 (B, C, D, H, W)
        gt_shape (tuple): 真实 Mask 的形状 (D, H, W)
    """
    if pred_seg.dim() == 3:
        input_tensor = pred_seg.unsqueeze(0).unsqueeze(0).float() # 插值需要 float 类型
    else:
        input_tensor = pred_seg.float()

    resized_pred = F.interpolate(input_tensor, size=gt_shape, mode='nearest')

    if pred_seg.dim() == 3:
        resized_pred = resized_pred.squeeze().long()
    
    return resized_pred

            
def run_inference(hparams: argparse.Namespace):
    """
    Main inference loop with batch_size=1 and default_collate.
    """
    fabric = Fabric(accelerator=hparams.accelerator, devices=hparams.devices)
    fabric.seed_everything(hparams.seed)

    ckpt_name = os.path.basename(hparams.checkpoint_path)
    ckpt_folder_name = os.path.splitext(ckpt_name)[0]
    final_output_dir = os.path.join(hparams.output_dir, ckpt_folder_name)
    
    if fabric.is_global_zero:
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"[Rank {fabric.global_rank}] Saving results to: {final_output_dir}")
    
    fabric.barrier()

    # --- Load Data ---
    val_dataset = ValTextSeg(
        img_dir=hparams.img_dir,
        gt_dir=hparams.gt_dir,
        image_size=hparams.image_size
    )
    
    # Enforce batch_size=1 and remove custom collate_fn
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=hparams.num_workers,
        collate_fn=None  # Use default_collate
    )

    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    model = TextSAM(
        image_encoder=None,  
        text_embedder=hparams.text_embedder_path,
        text_length=hparams.text_length,
        ds_scale=hparams.ds_scale,
        image_size=hparams.image_size,
    )

    if not hparams.checkpoint_path or not os.path.exists(hparams.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {hparams.checkpoint_path}")
    print(f"[Rank {fabric.global_rank}] Loading checkpoint from: {hparams.checkpoint_path}")
    # model.load_from_local(hparams.checkpoint_path, strict=False)
    
    model = fabric.setup(model)
    model.eval()
    
    # --- Inference Loop ---
    print(f"[Rank {fabric.global_rank}] Starting inference...")

    for batch in val_dataloader:
        images = batch['image']       # (1, D, H, W)
        prompt_dict = batch['prompt_dict'] # Dict of lists/tensors
        file_name = batch['img_name']
        org_spacing = batch['img_spacing']
        tar_spacing = batch['gt_spacing']
        gts = batch['mask']
        base_name = os.path.basename(file_name[0])
        
        input_images = images[0].unsqueeze(1).expand(-1, 3, -1, -1).to(fabric.device)
        prompt_data = []
        with torch.no_grad():
            keys = list(prompt_dict.keys())
            for k in keys:
                if k == "instance_label": continue
                
                text_val = prompt_dict[k]
                text = text_val[0] if isinstance(text_val, (list, tuple)) else text_val
                if isinstance(text, (list, tuple)): text = text[0] # double check
                
                text_embed = model.text_embedder(text)
                if len(text_embed.shape) > 2:
                    text_embed = text_embed[:, -1]
                text_embed = text_embed.unsqueeze(1) # (1, 1, C)
                
                prompt_data.append((int(k), text_embed))

            image_pe = model.prompt_encoder.get_dense_pe()
            for start_idx in range(0, D, 300):
                end_idx = min(start_idx + 300, D)
                current_batch_size = end_idx - start_idx
                img_chunk = input_images[start_idx:end_idx]
                image_embeddings_chunk = model.image_encoder(img_chunk)
                with torch.no_grad():
                    for k_int, text_embed in prompt_data:
                        text_embedding_batch = text_embed.repeat(current_batch_size, 1, 1)
                        
                        queries, keys, q_cls, k_cls = model.mask_former(image_embeddings_chunk, text_embedding_batch)
                    
        save_path = os.path.join(final_output_dir, base_name)
        np.savez_compressed(save_path,  q_cls=q_cls.squeeze().cpu().numpy(), k_cls=k_cls.squeeze().cpu().numpy())

    fabric.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextSAM Inference")
    torch.set_float32_matmul_precision('high')
    
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    
    parser.add_argument("--img_dir", default='/root/autodl-tmp/dataset/sample/img', type=str)
    parser.add_argument("--gt_dir", default='/root/autodl-tmp/dataset/sample/sample_gt', type=str)
    parser.add_argument("--checkpoint_path", default='/root/autodl-tmp/Rep-MedSAM/ckpts/epoch=0035-step=042000.ckpt', type=str)
    parser.add_argument("--output_dir", default='/root/autodl-tmp/dataset/sample/seg', type=str)
    
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--text-embedder-path", type=str, default="./ckpts/bert")
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--ds-scale", type=float, default=4.)

    hparams = parser.parse_args()
    run_inference(hparams)