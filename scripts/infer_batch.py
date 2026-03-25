import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from lightning.fabric import Fabric
import numpy as np
from matplotlib import pyplot as plt
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from training.train_text_seg import TextSAM
    from dataset.textseg import ValTextSeg, process_output
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
    result = model.load_from_local(hparams.checkpoint_path, strict=False)
    print(result)
    model = fabric.setup(model)
    model.eval()
    
    # --- Inference Loop ---
    print(f"[Rank {fabric.global_rank}] Starting inference...")

    for batch in val_dataloader:
        images = batch['image']       # (1, D, H, W)
        prompt_dict = batch['prompt_dict'] # Dict of lists/tensors
        file_name = batch['img_name']
        padded_size = batch["pad_s"]
        pad_width = batch["pad_w"]
        base_name = os.path.basename(file_name[0])
        
        input_images = images[0].unsqueeze(1).expand(-1, 3, -1, -1).to(fabric.device)
        D, _, H, W = input_images.shape
        final_seg = np.zeros((D, H, W), dtype=np.uint8)
        max_scores = np.full((D, H, W), -10000.0, dtype=np.float32) 
        
        prompt_embeddings = []
    prompt_class_ids = []

    with torch.no_grad():
        keys = sorted(list(prompt_dict.keys())) 
        for k in keys:
            if k == "instance_label": continue
            
            text_val = prompt_dict[k]
            text = text_val[0] if isinstance(text_val, (list, tuple)) else text_val
            if isinstance(text, (list, tuple)): text = text[0]
            
            text_embed = model.text_embedder(text)
            if len(text_embed.shape) > 2:
                text_embed = text_embed[:, -1]
            
            prompt_embeddings.append(text_embed)
            prompt_class_ids.append(int(k))
            
        all_prompts_tensor = torch.cat(prompt_embeddings, dim=0).to(fabric.device) 
        class_map = torch.tensor(prompt_class_ids, device=fabric.device, dtype=torch.long)
        num_prompts = len(prompt_class_ids)

        print(f"[Info] Total prompts: {num_prompts}. Class IDs: {prompt_class_ids}")

    
        image_pe = model.prompt_encoder.get_dense_pe()
        
        image_embeddings = []
        img_chunk_size = 20 # <--- [关键] 请根据显存调整这个值！
        for start_idx in range(0, D, img_chunk_size):
            end_idx = min(start_idx + img_chunk_size, D)
            current_bs = end_idx - start_idx
            
            # (B, 3, H, W)
            img_chunk = input_images[start_idx:end_idx] 
            
            # 1. Image Encoder (B, C, h, w)
            image_embeddings_chunk = model.image_encoder(img_chunk)
            image_embeddings.append(image_embeddings_chunk)
        image_embeddings = torch.cat(image_embeddings, dim=0) #NB, C, H, W
    
        text_chunk_size = 20
        for start_idx in range(0, D, text_chunk_size):
            end_idx = min(start_idx + text_chunk_size, D)
            bs = end_idx - start_idx
            image_embeddings_chunk = image_embeddings[start_idx: end_idx].unsqueeze(1).expand(-1, text_chunk_size, -1, -1, -1).flatten(0,1)
            print(image_embeddings_chunk.shape)
            queries, keys, q_cls, k_cls = model.mask_former(image_embeddings_chunk, all_prompts_tensor)
            
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None, boxes=None, masks=None, text=q_cls,
            )
            
            seg_chunk, _ = model.mask_decoder(
                image_embeddings=image_embeddings_chunk,
                image_pe=image_pe, 
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            logits_bn = seg_chunk.view(bs, text_chunk_size, 64, 64)
                


            max_vals, max_indices = torch.max(logits_bn, dim=1)
            
            # 4. 阈值过滤
            # 只有最高分 > 0 (即 prob > 0.5) 才算前景，否则是背景 (0)
            foreground_mask = max_vals > 0.0 # Logits > 0 等价于 Sigmoid > 0.5
            
            # 5. 映射回真实的 Class ID
            # 比如 max_indices 是 0, 对应的真实 ID 是 1 (Liver)
            pred_labels = class_map[max_indices] # (B, H, W)
            
            # 6. 应用背景掩膜
            # 如果没过阈值，置为 0 (背景)
            final_pred_chunk = pred_labels * foreground_mask.long()
            
            # 7. 填入最终大数组
            final_seg[start_idx:end_idx] = final_pred_chunk.cpu().numpy().astype(np.uint8)
            
            # 可选：如果你还需要保存 max_scores 用于调试
            max_scores[start_idx:end_idx] = max_vals.cpu().numpy()
                            
        padded_size = padded_size.item()
        pad_width = tuple(tuple(w.item() for w in x) for x in pad_width)
        save_path = os.path.join(final_output_dir, base_name)
        np.savez_compressed(save_path, segs=final_seg, padded_size=padded_size, pad_width=pad_width)

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
    parser.add_argument("--checkpoint_path", default='/root/autodl-tmp/Rep-MedSAM/ckpts/epoch=0028-step=034000.ckpt', type=str)
    parser.add_argument("--output_dir", default='/root/autodl-tmp/dataset/sample/seg', type=str)
    
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--text-embedder-path", type=str, default="./ckpts/bert")
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--ds-scale", type=float, default=4.)

    hparams = parser.parse_args()
    run_inference(hparams)