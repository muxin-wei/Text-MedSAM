import os
import os.path as osp
import json
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from utils.helper import instantiate_from_config
import torchvision.utils as vutils

def get_color_overlay(img, pred, gt):
    if pred.ndim == 3:
        pred = np.squeeze(pred)
    if gt.ndim == 3:
        gt = np.squeeze(gt)
   
    if pred.ndim > 2:
        pred = pred[0]
    if gt.ndim > 2:
        gt = gt[0]
    h, w = gt.shape[:2]
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
   
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    res = img.copy()
    res[(gt == 1) & (pred == 0)] = [0, 255, 0]
    res[(gt == 0) & (pred == 1)] = [255, 0, 0]
    res[(gt == 1) & (pred == 1)] = [255, 255, 0]
   
    return cv2.addWeighted(res, 0.7, img, 0.3, 0)


def add_zoomin_inset(vis_img, ref_mask, pixel_threshold=1500, inset_size=160, margin=20, box_color=(255, 255, 255), thickness=2):
    area = np.sum(ref_mask)
    print(f"Debug - Mask area: {area}")
    
    if area > 0:
        vis_img_out = vis_img.copy()
        h, w = vis_img.shape[:2]
        y_indices, x_indices = np.where(ref_mask > 0)
       
        if len(y_indices) == 0 or len(x_indices) == 0:
            return vis_img
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min = max(0, y_min - margin)
        y_max = min(h, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(w, x_max + margin)
        crop = vis_img_out[y_min:y_max, x_min:x_max].copy()
       
        if crop.size == 0:
            return vis_img
        crop_h, crop_w = crop.shape[:2]
        max_side = max(crop_h, crop_w)
        square_crop = np.zeros((max_side, max_side, 3), dtype=np.uint8)
       
        y_off = (max_side - crop_h) // 2
        x_off = (max_side - crop_w) // 2
        square_crop[y_off:y_off+crop_h, x_off:x_off+crop_w] = crop
        inset = cv2.resize(square_crop, (inset_size, inset_size), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(inset, (0, 0), (inset_size-1, inset_size-1), box_color, thickness)
        
        pad = 15
        y_start = h - inset_size - pad
        y_end = h - pad
        x_start = pad         
        x_end = pad + inset_size
        vis_img_out[y_start:y_end, x_start:x_end] = inset
        return vis_img_out
    return vis_img

config = OmegaConf.load("/root/autodl-tmp/Text-MedSAM/logs/19_epoch/configs/2026-03-04T10-37-20_text_seg_new-project.yaml")
model = instantiate_from_config(config.model)
device = torch.device("cuda")
model.load_from_local("/root/autodl-tmp/Text-MedSAM/logs/19_epoch/checkpoints/epoch=0019-step=075000.ckpt", strict=True)
model.to(device).eval()

micro = "pet"
val_dir = f"/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/test/val_2d_{micro}/img"
prompt_dir = f"/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/test/val_2d_{micro}/prompts"
save_seg_dir = val_dir.replace("img", "pet_attn_seg")
os.makedirs(save_seg_dir, exist_ok=True)
files = sorted(glob(osp.join(val_dir, "*.npy"), recursive=True))

@torch.no_grad()
def inference(model, imgs, texts):
    img_embed, feats = model.encode_image(imgs)
    text_embeddings = model.text_embedder(text=texts, mode="", ds_ids=None, c_ids=None)
    feats = feats[-2::-1]
    if text_embeddings.ndim == 2:
        text_embeddings = text_embeddings.unsqueeze(1)
    elif text_embeddings.ndim == 3 and text_embeddings.shape[1] != 1:
         text_embeddings = text_embeddings[:, 0:1, :]

    output, text_out, attn_weights, out_x = model.mask_former(img_embed, text_embeddings, feats)
    logits = model.mask_decoder(output, text_embeddings)
    attn = []
    for out in out_x:
        attn.append(model.mask_decoder(out, text_embeddings))
   
    return logits, attn, feats, attn_weights

with torch.no_grad():
    for f in tqdm(files, desc="Processing Images"):
        try:
            base_filename = osp.splitext(osp.basename(f))[0]
            gt_path = f.replace("img", "gts")
            gt = np.load(gt_path, allow_pickle=True)
            img_np = np.load(f, allow_pickle=True)

            json_path = osp.join(prompt_dir, f"{base_filename}.json")
            if not osp.exists(json_path):
                continue
           
            with open(json_path, 'r', encoding='utf-8') as pf:
                prompt_dict = json.load(pf)
           
            is_instance = False
            if "is_instance" in prompt_dict:
                is_instance = bool(int(prompt_dict.pop("is_instance")))
            target_ids = []
            text_prompts = []
            for lid_str, text_str in prompt_dict.items():
                target_ids.append(int(lid_str))
                text_prompts.append(text_str)
           
            N = len(text_prompts)
            img_tensor = torch.from_numpy(img_np).float()
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            elif img_tensor.ndim == 3:
                if img_tensor.shape[2] <= 4:
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                else:
                    img_tensor = img_tensor.unsqueeze(0)
           
            if img_tensor.max() > 1.0:
                img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-6)
           
            img_tensor = img_tensor.to(device)
            batch_img_tensor = img_tensor.repeat(N, 1, 1, 1)
           
            logits, attn_weights, feats, text_out = inference(model, batch_img_tensor, text_prompts)
           
            pred_mask = torch.sigmoid(logits)
            if pred_mask.ndim == 4:
                pred_mask = pred_mask.squeeze(1)
            pred_mask_np = pred_mask.cpu().numpy()
            pred_binary_batch = (pred_mask_np > 0.5).astype(np.uint8)
            print(attn_weights[0].shape)
            
            if isinstance(attn_weights, list):
                target_attns = attn_weights[-3:]
            else:
                target_attns = [attn_weights]
           
            attn_maps_list_batch = []
            for attn in target_attns:
                if attn.ndim == 4:
                    attn = attn.squeeze(1)
                attn_maps_list_batch.append(attn.cpu().numpy())
            
            bbox_raw = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            
            ZOOM = 999999999   # 强制触发
            
            for i in range(N):
                current_id = target_ids[i]
                current_text = text_prompts[i]
               
                pred_bi = pred_binary_batch[i]
               
                if is_instance:
                    gt_bi = (gt > 0).astype(np.uint8)
                else:
                    gt_bi = (gt == current_id).astype(np.uint8)
               
                gt_h, gt_w = gt_bi.shape[:2]
               
                img_gray_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img_rgb = cv2.cvtColor(img_gray_norm, cv2.COLOR_GRAY2RGB)
               
                if img_rgb.shape[:2] != (gt_h, gt_w):
                    img_rgb = cv2.resize(img_rgb, (gt_w, gt_h))
               
                text_gt_show = get_color_overlay(img_rgb.copy(), pred_bi.copy(), gt_bi.copy())
                
                # <<< 关键修复：GT=0 时用 Pred 补上，确保 zoomin 触发（复用 gt_bi 变量，不改名！）
                # gt_bi = np.logical_or(gt_bi, pred_bi).astype(np.uint8)
                
                pred_show_zoom = add_zoomin_inset(text_gt_show.copy(), gt_bi.copy(), pixel_threshold=ZOOM)
                
                heatmaps_show = []
                attn_layers_to_show = attn_maps_list_batch[-3:]
               
                for idx, attn_map_batch_layer in enumerate(attn_layers_to_show):
                    attn_map = attn_map_batch_layer[i]
                   
                    if attn_map.shape[:2] != (gt_h, gt_w):
                        attn_map = cv2.resize(attn_map, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
                       
                    denom = attn_map.max() - attn_map.min() + 1e-8
                    attn_norm = (attn_map - attn_map.min()) / denom
                   
                    heatmap_color_bgr = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_MAGMA)
                    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
                   
                    if idx == len(attn_layers_to_show) - 1:
                        mask_heat = (attn_norm > 0.1).astype(np.float32)
                        mask_heat = np.stack([mask_heat] * 3, axis=-1)
                        heatmap_show_rgb = (heatmap_color_rgb * mask_heat * 0.5 + img_rgb.copy() * (1 - mask_heat * 0.5)).astype(np.uint8)
                        heat_show_zoom = add_zoomin_inset(heatmap_show_rgb.copy(), gt_bi.copy(), pixel_threshold=ZOOM)
                        heatmaps_show.append(heat_show_zoom)
                    else:
                        heatmaps_show.append(heatmap_color_rgb)
                
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
               
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Input\n{current_text}", fontsize=14)
                axes[0].axis('off')
               
                if len(heatmaps_show) > 0:
                    axes[1].imshow(heatmaps_show[0])
                    axes[1].set_title("Attention (Layer -3)", fontsize=14)
                axes[1].axis('off')
               
                if len(heatmaps_show) > 1:
                    axes[2].imshow(heatmaps_show[1])
                    axes[2].set_title("Attention (Layer -2)", fontsize=14)
                axes[2].axis('off')
               
                if len(heatmaps_show) > 2:
                    axes[3].imshow(heatmaps_show[2])
                    axes[3].set_title("Attention (Layer -1)", fontsize=14)
                axes[3].axis('off')
               
                axes[4].imshow(pred_show_zoom)
                axes[4].set_title("Pred", fontsize=14)
                axes[4].axis('off')
               
                plt.tight_layout()
                save_name = osp.join(save_seg_dir, f"{base_filename}_target_{current_id}.png")
                plt.savefig(save_name, bbox_inches='tight', dpi=300)
                plt.close(fig)
               
        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback
            traceback.print_exc()
            continue