import os
import glob
import json
import numpy as np
from tqdm import tqdm

def analyze_non_empty_slices(data_dir: str, output_json: str):
    """
    Read all 3D .npz files, extract depth and indices of non-empty 'gts' slices,
    and save the statistics to a JSON file.
    """
    npz_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
    print(f"[*] Found {len(npz_files)} .npz files in {data_dir}")
    
    slice_info_dict = {}
    
    for file_path in tqdm(npz_files, desc="Analyzing Slices"):
        filename = os.path.basename(file_path)
        
        try:
            # Load the npz file
            data = np.load(file_path, allow_pickle=True)
            
            if 'gts' not in data:
                print(f"[!] Warning: 'gts' not found in {filename}, skipping.")
                continue
                
            gts = data['gts']
            
            # Ensure the array is 3D: (D, H, W)
            if gts.ndim != 3:
                print(f"[!] Warning: {filename} 'gts' is not 3D (shape {gts.shape}), skipping.")
                continue
                
            D, H, W = gts.shape
            
            # np.any is faster and safer than np.sum for checking non-zero pixels
            # axis=(1, 2) checks across H and W for each slice D
            non_empty_mask = np.any(gts != 0, axis=(1, 2))
            
            # Convert boolean mask to list of indices
            non_empty_indices = np.where(non_empty_mask)[0].tolist()
            
            # Store metadata
            slice_info_dict[filename] = {
                "depth": D,
                "num_non_empty": len(non_empty_indices),
                "non_empty_slices": non_empty_indices
            }
            
        except Exception as e:
            print(f"[-] Error processing {filename}: {e}")
            
    # Save results to a formatted JSON file
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(slice_info_dict, f, indent=4)
        
    print(f"[+] Analysis complete! Saved to {output_json}")

if __name__ == "__main__":
    DATA_DIR = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/train_npz_256"
    OUTPUT_JSON = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/slice_info.json"
    
    analyze_non_empty_slices(DATA_DIR, OUTPUT_JSON)