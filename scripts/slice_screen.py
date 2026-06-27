import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def build_label_dicts(meta_json):
    valid_label_dict = {}
    is_instance_dict = {}
    
    with open(meta_json, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        for k in meta_data.keys():
            # get the valid class id set
            valid_label_dict[k] = set([
                int(v) for v in meta_data[k] if v != "instance_label"
            ])
            is_instance_dict[k] = bool(meta_data[k].get("instance_label", False))
            
    return valid_label_dict, is_instance_dict

def validate_slice(task_args):
    """
    check each slice
    return: (is_valid, gt_path, img_path, reason)
    """
    gt_path, img_path, ds_name, valid_labels, is_instance = task_args
    
    try:
        gts = np.load(gt_path, mmap_mode='r', allow_pickle=True)
        present_ids = np.unique(gts)
        present_ids = present_ids[present_ids > 0] # rule out bg mask
        
        if not is_instance:
            valid_ids = [k for k in present_ids if k in valid_labels]
        else:
            valid_ids = present_ids
            
        if len(valid_ids) < 1:
            return (False, gt_path, img_path, f"No valid labels. Found: {present_ids}")
        return (True, gt_path, img_path, "Valid")
        
    except Exception as e:
        return (False, gt_path, img_path, f"Read Error: {str(e)}")

def clean_dataset(data_dir, gts_dir, meta_json, dry_run=True, num_workers=32):
    print("=== Dataset Cleaner Initializing ===")
    print(f"Dry Run Mode: {'ON (No files will be deleted)' if dry_run else 'OFF (DANGER: Files will be deleted)'}")
    
    valid_label_dict, is_instance_dict = build_label_dicts(meta_json)
    
    print("\nScanning for GT files...")
    tasks = []
    
    for ds_name in os.listdir(data_dir):
        full_path = osp.join(data_dir, ds_name)
        
        if osp.isdir(full_path) and ds_name in valid_label_dict:
            files = [f for f in os.listdir(full_path) if osp.isfile(osp.join(full_path, f))]
            
            for file in files:
                if file.endswith(".npy"):
                    img_path = osp.join(full_path, file)
                    gt_path = osp.join(gts_dir, file) 
                    
                    if not osp.exists(gt_path):
                        print(f"[Warning] GT missing for {file}, skipping...")
                        continue
                    
                    tasks.append((
                        gt_path, 
                        img_path, 
                        ds_name, 
                        valid_label_dict[ds_name], 
                        is_instance_dict[ds_name]
                    ))

    print(f"Found {len(tasks)} slices to validate. Starting multi-threaded check...")
    
    # 2. multithreading
    invalid_files = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(validate_slice, tasks), total=len(tasks), desc="Validating"))
        
    for res in results:
        is_valid, gt_p, img_p, reason = res
        if not is_valid:
            invalid_files.append((gt_p, img_p, reason))

    print("\n=== Validation Complete ===")
    print(f"Total slices: {len(tasks)}")
    print(f"Invalid slices: {len(invalid_files)}")
    
    if len(invalid_files) == 0:
        print("Dataset is perfectly clean!")
        return
        

    with open("invalid_PET.txt", 'w') as f:
        for gt_p, img_p, reason in invalid_files:
            f.write(img_p+'\n')
            
    if dry_run:
        print("\n[DRY RUN] The following files WOULD be deleted:")
        for gt_p, img_p, reason in invalid_files[:10]:
            print(f" - {osp.basename(gt_p)} | Reason: {reason}")
        if len(invalid_files) > 10:
            print(f"   ... and {len(invalid_files) - 10} more.")
        print("\nTo actually delete these files, run with dry_run=False.")
    else:
        print("\n[EXECUTING DELETION]")
        deleted_count = 0
        for gt_p, img_p, _ in tqdm(invalid_files, desc="Deleting files"):
            try:
                if osp.exists(gt_p):
                    os.remove(gt_p)
                if osp.exists(img_p):
                    os.remove(img_p)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {osp.basename(gt_p)}: {e}")
        print(f"Successfully deleted {deleted_count} pairs of invalid files.")

if __name__ == "__main__":
    IMG_DIR = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/train_npz_256/Microscopy"
    GTS_DIR = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/gts"
    META_JSON = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json"
    
    clean_dataset(IMG_DIR, GTS_DIR, META_JSON, dry_run=True, num_workers=48)