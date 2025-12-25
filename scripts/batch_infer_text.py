import argparse
import os
import sys
import torch
import torch.nn.functional as F
from typing import OrderedDict
import numpy as np
from glob import glob
import os.path as osp
from tqdm import tqdm
import random
import cv2
from datetime import datetime
from time import time
import cc3d
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from training.train_textsam import TextSAM
    from utils.utils import replace_batchnorm
    from src.models.visual.image_encoder import repvit_m1_0
except ImportError as e:
    print(f"Error importing model/dataset classes: {e}")
    print("Please ensure textsam/train_text_seg.py and textsam/textseg.py are accessible.")
    sys.exit(1)


def pad_image_3d(image_3d, target_size=256, pad_val=0):
    """
    padding volumes on H, W。
    inputs: (D, H, W)
    """
    D, h, w = image_3d.shape
    padh = target_size - h
    padw = target_size - w
    
    padding = ((0, 0), (0, padh), (0, padw))
    
    image_padded = np.pad(image_3d, padding, mode='constant', constant_values=pad_val)
    
    return image_padded

@torch.no_grad()
def postprocess_masks(masks, new_size, original_size):
    """
    Do cropping and resizing

    Parameters
    ----------
    masks : torch.Tensor
        masks predicted by the model
    new_size : tuple
        the shape of the image after resizing to the longest side of 256
    original_size : tuple
        the original shape of the image

    Returns
    -------
    torch.Tensor
        the upsampled mask to the original size
    """
    # Crop
    masks = masks[..., :new_size[0], :new_size[1]]
    # Resize
    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )

    return masks

@torch.no_grad()
def keep_largest_component(mask):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    label_out, N = cc3d.connected_components(mask_np, connectivity=26, return_N=True)
    if not N:
        return mask
    stats = cc3d.statistics(label_out)
    vox_count = stats['voxel_counts'][1:]
    if len(vox_count) == 0:
        return mask
    largest = np.argmax(vox_count) + 1
    mask_np = (label_out ==)

@torch.no_grad()
def run_inference(img_npz_file, model):
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    
    img_3d = npz_data['imgs']
    prompt_dict = npz_data['text_prompts'].tolist()
    D, H, W = img_3d.shape
    img_256 = resize_longest_side_3d(img_3d, 256)
    new_h, new_w = img_256.shape[-2:]
    img_256 =  pad_image_3d(img_256)

    img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_max=None, a_min=1e-8
        )
    images = torch.from_numpy(img_256).float().to(model.device)
    images = images.unsqueeze(1).expand(-1, 3, -1, -1)
    image_pe = model.prompt_encoder.get_dense_pe()
    image_chunk_size = 280
    image_embeddings = []
    for start_idx in range(0, D, image_chunk_size):
        end_idx = min(start_idx + image_chunk_size, D)
        image_chunk = images[start_idx:end_idx]
        chunk_embdding = model.image_encoder(image_chunk)
        image_embeddings.append(chunk_embdding)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    all_k_seg = []
    label_ids_list = []
    sorted_keys = sorted(list(k for k in prompt_dict.keys() if k != 'instance_label'))
    if not sorted_keys:
        sorted_keys = [str(prompt_dict['instance_label'])]
    
    for k in sorted_keys: # prompts
        text_input = prompt_dict[k]
        label_ids_list.append(int(k))
        text_embed = model.text_embedder(text_input)
        if len(text_embed.shape) < 2:
            text_embed = text_embed.unsqueeze(1)
        elif len(text_embed) > 2 and text_embed.shape[1] > 1:
            text_embed = text_embed[:,0]
        k_logit_chunks = []
        for start_idx in range(0, D, image_chunk_size): # chunks
            end_idx = min(start_idx + image_chunk_size, D)
            chunk_embdding = image_embeddings[start_idx : end_idx]
            text_embed_repeat = text_embed.repeat(chunk_embdding.shape[0], 1, 1)
            queries, keys, q_cls, k_cls = model.mask_former(chunk_embdding, text_embed_repeat)
            sparse_embeddings, dense_embeddings = model.prompt_encoder( 
                points=None,
                boxes=None,
                masks=None,
                text=q_cls,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=chunk_embdding,
                image_pe=image_pe, 
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=queries,
                multimask_output=False,
            )
            k_logit_chunks.append(low_res_masks) # chunk_seg
        k_res_pred = torch.cat(k_logit_chunks, dim=0).squeeze().unsqueeze(1)# prompt_seg
        k_res_pred = postprocess_masks(k_res_pred, new_size=[new_h, new_w], original_size=[H, W])
        k_res_pred = torch.sigmoid(k_res_pred).squeeze()
        all_k_seg.append(k_res_pred)
    
    segs = torch.stack(all_k_seg, dim=0) # K, D, H, W
    segs, max_ids = torch.max(segs, dim=0, keepdim=True) 
    
    binary_seg = segs > 0.5 # 
    out_seg = torch.zeros_like(binary_seg[0],dtype=torch.uint8)
    label_map = torch.tensor(label_ids_list, dtype=torch.uint8, device=max_ids.device)
    cc_mask = []
    for k in range(binary_seg.shape[0]):
        raw_mask = binary_seg[k]
        cleaned_mask = keep_largest_component(raw_mask)
    out_labels = label_map[max_ids]
    out_seg[mask] = out_labels[mask].to(torch.uint8)

    base_name = os.path.basename(img_npz_file)
    save_path = os.path.join(save_dir, base_name)
    np.savez_compressed(save_path, segs=out_seg.squeeze().cpu().numpy())
    
    # all_logits = torch.cat(all_k_logits, dim=0).squeeze()
    # max_scores, max_indices = torch.max(all_logits, dim=0)
    # segs = torch.zeros((D, H, W), dtype=torch.uint8, device=model.device)
    
    # for idx, real_label_id in enumerate(label_ids_list):
    #         mask = (max_indices == idx) & (max_scores > 0.5)
    #         segs[mask] = real_label_id


    # np.savez_compressed(save_path, segs=segs.squeeze().cpu().numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextSAM 3D Medical Segmentation Inference Script")
    torch.set_float32_matmul_precision('high')
    # DDP/System Args
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    
    # Data/Model Args
    parser.add_argument("--img_dir", default='/root/autodl-tmp/dataset/sample/img', type=str,  help="Path to validation images")
    parser.add_argument("--checkpoint_path", default='/root/autodl-tmp/Rep-MedSAM/ckpts/epoch=0035-step=042000.ckpt', type=str, help="Path to checkpoint")
    parser.add_argument("--output_dir", default='/root/autodl-tmp/dataset/sample/seg', type=str, help="Base directory to save results")
    
    hparams = parser.parse_args()
    
    os.environ["PL_GLOBAL_SEED"] = str(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    ckpt_name = os.path.basename(hparams.checkpoint_path)
    ckpt_folder_name = os.path.splitext(ckpt_name)[0]
    
    model = TextSAM(
        image_encoder=repvit_m1_0(),
        text_embedder="/root/autodl-tmp/Rep-MedSAM/ckpts/bert",
        text_length=256,
        ds_scale=4.0,
        image_size=256,
    )
    replace_batchnorm(model.image_encoder)

    if not hparams.checkpoint_path or not os.path.exists(hparams.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {hparams.checkpoint_path}")
    print(f"Loading checkpoint from: {hparams.checkpoint_path}")
    model.load_from_local(hparams.checkpoint_path, strict=False)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    
    save_dir = os.path.join(hparams.output_dir, ckpt_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    img_npz_files = sorted(glob(osp.join(hparams.img_dir, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    
    for img_npz_file in tqdm(img_npz_files[:]):
        start_time = time()
        run_inference(img_npz_file, model=model)
        end_time = time()
        efficiency['case'].append(osp.basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', osp.basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(osp.join(save_dir, 'efficiency.csv'), index=False)