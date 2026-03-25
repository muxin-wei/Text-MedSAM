import os
import random
from glob import glob
import os.path as osp

# --- Configuration ---
seed = 1234
data_dir = '/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/3D_val_npz/'
split_rate = 0.5
train_txt = '/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/val50p.txt'

random.seed(seed)
train_lines = []
val_lines = []

current_files = sorted(glob(osp.join(data_dir, "*.npz")))
random.shuffle(current_files)

# 按比例计算 split
split_idx = int(len(current_files) * split_rate)

# 分配到对应的列表
for i, f in enumerate(current_files):
    if i < split_idx:
        train_lines.append(f)
    else:
        val_lines.append(f)

# # 第一层遍历：数据集级别 (e.g., Dataset_A, Dataset_B)
# for ds_name in sorted(os.listdir(data_dir)):
#     current_files = sorted(glob(osp.join(sub_path, "*.npz")))
    
    
#     # 第二层遍历：子类或模态级别 (e.g., CT, MRI 或 Positive, Negative)
#     for sub_name in sorted(os.listdir(ds_path)):
#         sub_path = osp.join(ds_path, sub_name)
#         if not osp.isdir(sub_path):
#             continue
            
#         # 获取该二级目录下所有的 .npy 文件
#         current_files = sorted(glob(osp.join(sub_path, "*.npz")))
        
#         if not current_files:
#             continue
            
#         # 在当前目录下进行 Shuffle 
#         random.shuffle(current_files)
        
#         # 按比例计算 split
#         split_idx = int(len(current_files) * split_rate)
        
#         # 分配到对应的列表
#         for i, f in enumerate(current_files):
#             if i < split_idx:
#                 train_lines.append(f)
#             else:
#                 val_lines.append(f)
        
#         print(f"Processed: {ds_name}/{sub_name} | Found: {len(current_files)} files")

print("-" * 30)
print(f"Total training samples: {len(train_lines)}")
print(f"Total validation samples: {len(val_lines)}")

# --- Write to file ---
with open(train_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_lines) + '\n')

print(f"Successfully saved to {train_txt}")