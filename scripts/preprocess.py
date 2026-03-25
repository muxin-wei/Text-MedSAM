import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

def resize_3d_volume(input_dir: str, output_dir: str, img_size: int = 256):
    """
    Resize 'imgs' and 'gts' to (D, img_size, img_size) while preserving all other keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(input_dir, "**/*.npz"), recursive=True)
    print(f"[*] Found {len(npz_files)} files in {input_dir}")
    
    for file_path in tqdm(npz_files, desc="Processing Volumes"):
        relpath = os.path.relpath(file_path, input_dir)
        file_dir = os.path.join(output_dir, os.path.dirname(relpath))
        os.makedirs(file_dir, exist_ok=True)
        out_path = os.path.join(output_dir, relpath)
        
        # allow_pickle=True is sometimes needed if metadata contains strings/dicts
        data = np.load(file_path, allow_pickle=True)
        
        out_dict = {}
        
        # 1. Process 'imgs' and 'gts'
        if 'imgs' in data and 'gts' in data:
            imgs = data['imgs']
            gts = data['gts']
            D, H, W = imgs.shape
            
            if H == img_size and W == img_size:
                out_dict['imgs'] = imgs
                out_dict['gts'] = gts
            else:
                new_imgs = np.zeros((D, img_size, img_size), dtype=imgs.dtype)
                new_gts = np.zeros((D, img_size, img_size), dtype=gts.dtype)
                
                # Resize slice by slice
                for d in range(D):
                    img_slice = imgs[d].astype(np.float32)
                    new_imgs[d] = cv2.resize(img_slice  , (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    new_imgs[d] = new_imgs[d].astype(imgs.dtype)
                    # Nearest neighbor for segmentation masks
                    gt_slice = gts[d].astype(np.float32)
                    new_gts[d] = cv2.resize(gt_slice, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    new_gts[d] = new_gts[d].astype(gts.dtype)
                    
                out_dict['imgs'] = new_imgs
                out_dict['gts'] = new_gts
        
        # 2. Copy all other original key-value pairs
        for key in data.files: 
            if key not in ['imgs', 'gts']:
                out_dict[key] = data[key]
        np.savez_compressed(out_path, **out_dict)
        os.remove(file_path)
        
if __name__ == "__main__":
    INPUT_DIR = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/3D_train_npz_all"
    OUTPUT_DIR = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/train_npz_256"
    TARGET_SIZE = 256
    
    resize_3d_volume(INPUT_DIR, OUTPUT_DIR, img_size=TARGET_SIZE)
    print("[+] Preprocessing completed!")