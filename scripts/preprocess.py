import glob
import numpy as np
import os
import json
import os.path as osp

data_list = []
data_dir = '/root/autodl-tmp/dataset/train'
with open("/root/autodl-tmp/dataset/text_seg_class.json", 'r') as f:
    text_labels = json.load(f)
    
valid_keys = set(text_labels.keys())
npz_files = glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
npz_files = sorted([
    file_path 
    for file_path in npz_files
    if osp.basename(osp.dirname(file_path)) in valid_keys
])

for file_path in npz_files:
    try:
        data = np.load(file_path, mmap_mode='r')
        n_slice = data['imgs'].shape[0]
        dataset_name = os.path.basename(os.path.dirname(file_path))
        data_list.append({
            "file_path": file_path,
            "n_slice": n_slice,
            "dataset": dataset_name
        })
    except:
        pass
    
output_path = os.path.join(data_dir, "meta.json")
with open(output_path, "w") as f:
    json.dump(data_list, f, indent=4)