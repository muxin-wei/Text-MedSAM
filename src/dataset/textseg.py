import os
import os.path as osp
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .utils import pad_image_3d, resize_longest_side_3d, resize_longest_side_2d, pad_image_2d

class SliceTextSeg(Dataset):
    def __init__(self, data_dir, gts_dir, text_embed, meta_json):
        with open(data_dir, 'r') as f:
            self.npy_files = [line.strip() for line in f.readlines() if line.strip()]
        self.gts_dir = gts_dir
        info = torch.load(text_embed, map_location='cpu')
        self.dataset2id = info['dataset2id']
        self.class2id = info['class2id']
        self.is_instance_dict = {}
        self.valid_label_dict = {}

        with open(meta_json, 'r') as f:
            meta_data = json.load(f)
            for k in meta_data.keys():
                self.valid_label_dict[k] = set([int(v) for v in meta_data[k] if v != "instance_label"])
                self.is_instance_dict[k] = bool(meta_data[k]["instance_label"])
                
    def __len__(self):
        return int(len(self.npy_files))

    def __getitem__(self, idx):
        file_path = self.npy_files[idx]
        filename = os.path.basename(file_path)
        img = np.load(file_path, mmap_mode='r', allow_pickle=True).astype(np.float32)
        gts = np.load(osp.join(self.gts_dir, filename), mmap_mode='r', allow_pickle=True)
        img_np = (img - img.min())  / (img.max() - img.min() + 1e-6)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        ds_name = osp.basename(osp.dirname(file_path))
        ds_id = self.dataset2id[ds_name]
        is_instance = self.is_instance_dict.get(ds_name, False)
        
        # segment target sample
        present_ids = np.unique(gts)
        present_ids = present_ids[present_ids > 0]
        if not is_instance:
            valid_ids = [k for k in present_ids if k in self.valid_label_dict[ds_name]]
        else:
            valid_ids = present_ids
        if len(valid_ids) < 1:
            print(f"{ds_name} --- {filename}\n valid: {self.valid_label_dict[ds_name]}\n exists:{present_ids}")
        target_id = random.choice(valid_ids)
        target_mask = torch.from_numpy(gts == target_id).float()
        cls_label = "1" if is_instance else str(target_id)
        c_id = self.class2id[ds_id].get(cls_label, 0)
        
        target_mask = target_mask.unsqueeze(0)
        
        return{
            "image": img_tensor,
            "mask": target_mask,
            "cls_id": c_id,
            "ds_id": ds_id,
        }
        
class TextSeg(Dataset):
    def __init__(self, data_dir, text_label_path, image_size=256,  n_slicing=3, max_instances=5):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g., dataset/train_10/)
            text_label_path (str): Path to the text label JSON file.
            image_size (int): Target image size for resizing (default: 256).
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.n_slicing = n_slicing
        with open(text_label_path, 'r') as f:
            self.text_labels = json.load(f)
        valid_keys = set(self.text_labels.keys())
        self.samples = glob.glob(osp.join(data_dir, '**/*.npz'),recursive=True)
        self.samples = sorted([
            file_path 
            for file_path in self.samples 
            if osp.basename(osp.dirname(file_path)) in valid_keys
        ])
        self.max_instances = max_instances
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        dataset_name = osp.basename(osp.dirname(file_path))
        try:
            data = np.load(file_path)
            imgs = data['imgs']
            mask = data['gts']
            text_prompt = self.text_labels[dataset_name]
        except Exception as e:
            print(f'error loading {file_path}: {e}')
            return None
        D, H, W = imgs.shape
        if D < self.n_slicing:
            num_pad = self.n_slicing - D
            indices = np.concatenate([np.arange(D), np.full(num_pad, D - 1)]).astype(int)
        else:
            indices = np.random.choice(np.arange(D), size=self.n_slicing, replace=False)
        images = imgs[indices].astype(np.uint8)
        masks = mask[indices].astype(np.uint8)
        
        images = pad_image_3d(resize_longest_side_3d(images, target_length=self.image_size, mode=cv2.INTER_CUBIC)) 
        masks = pad_image_3d(resize_longest_side_3d(masks, target_length=self.image_size, mode=cv2.INTER_NEAREST))
        images, masks = random_transform(images, masks)
        
        valid_keys = [int(k) for k in text_prompt.keys() if k.isdigit()]
        is_instance = text_prompt.get('instance_label') == 1
        batch_masks = np.zeros((self.n_slicing, self.max_instances, self.image_size, self.image_size), dtype=np.uint8)
        mask_ids = np.zeros((self.n_slicing * self.max_instances), dtype=np.uint8)
        pter = 0
        prompts = []
        
        for i in range(masks.shape[0]):
            slice = masks[i]
            unique_ids = [v for v in np.unique(slice) if v > 0]
            slice_prompts = []
            if len(unique_ids) > self.max_instances:
                selected_ids = np.random.choice(unique_ids, size=self.max_instances, replace=False)
            else:
                selected_ids = unique_ids
            np.random.shuffle(selected_ids)
            for k, cls_id in enumerate(selected_ids): # cls_id in slice
                batch_masks[i, k] = (slice == cls_id).astype(np.uint8)
                text_key = str(cls_id) if not is_instance else str(valid_keys[0])
                mask_ids[pter] = cls_id
                pter += 1 
                slice_prompts.append(random.choice(text_prompt[text_key]))
            while len(slice_prompts) < self.max_instances:
                slice_prompts.append("background")
                mask_ids[pter] = 0
                pter += 1
            prompts.extend(slice_prompts)
        
        batch_masks = batch_masks.reshape((-1, self.image_size, self.image_size))
        masks = np.stack(batch_masks, axis=0) if isinstance(batch_masks, list) else batch_masks
        text_prompt_sep = "[SEP]".join(prompts)
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        img_tensor = images.float() / 255.0
        
        return{
            "image": img_tensor.unsqueeze(1), # n_slices, 1, h, w
            "mask": torch.from_numpy(masks[:, np.newaxis, ...]).to(torch.long), # n*m, h, w
            "mask_ids": mask_ids,
            "text": text_prompt_sep,
            "img_name": osp.basename(file_path).split('.npz')[0]
        }


class TextSegVal(Dataset):
    def __init__(self, gts_dir, meta_json=None, image_size=256):
        """
        meta_json: .json file with information of valid slices for images
        """
        self.gts_dir = gts_dir
        self.image_size = image_size
        self.slice_map = []

        if meta_json and os.path.exists(meta_json):
            print(f"loading from : {meta_json}")
            with open(meta_json, 'r') as f:
                self.slice_map = json.load(f)

                    
    def __len__(self):
        return len(self.slice_map)
    
    def __getitem__(self, idx): 
        meta = self.slice_map[idx]
        file_path = meta["file_path"]
        s_idx = meta["slice_idx"]
        
        img_data = np.load(file_path, allow_pickle=True)
        gt_path = file_path.replace("3D_val_npz", "3D_val_gt/3D_val_gt_text")
        gt_data = np.load(gt_path)
        
        imgs = img_data['imgs'][s_idx]  # (H, W)
        gts = gt_data['gts'][s_idx]     # (H, W)
        h, w = imgs.shape[-2:]
        text_prompt = img_data["text_prompts"].tolist()
        imgs_res = resize_longest_side_2d(imgs, target_length=256, mode=cv2.INTER_CUBIC)
        gts_res  = resize_longest_side_2d(gts,  target_length=256, mode=cv2.INTER_NEAREST)
        
        images = pad_image_2d(imgs_res)   # (D, 256, 256)
        masks  = pad_image_2d(gts_res)    # (D, 256, 256)
        
        pad_info = {
            "original_shape": (h, w),   # (D_orig, H_orig, W_orig)
            "padded_shape": images.shape,       # (D, 256, 256)
            "image_name": osp.basename(file_path).split('.npz')[0],
            "file_path": file_path
        }
        
        images = torch.from_numpy(images).float()         # (D, 256, 256)
        images = (images - images.min()) / (images.max() - images.min() + 1e-6)
        images = images.view(1, 1, 256, 256).expand(-1, 3, -1, -1)         # (D, 3, 256, 256)
        masks = torch.from_numpy(masks).unsqueeze(0).to(torch.uint8)  # (D, 1, 256, 256)
        
        valid_keys = sorted([int(k) for k in text_prompt.keys() if k.isdigit()])
        is_instance = text_prompt.get('instance_label') == 1
        
        all_prompts = []   
        prompt_class_ids = []  
        
        if is_instance:
                all_prompts.append(text_prompt[str(valid_keys[0])])
                prompt_class_ids.append(1)
        else:
            for cls_id in valid_keys:
                all_prompts.append(text_prompt[str(cls_id)])
                prompt_class_ids.append(int(cls_id))
        return {
            "image": images,
            "mask": masks,
            "all_prompts": " [SEP] ".join(all_prompts), 
            "prompt_class_ids": ",".join(map(str, prompt_class_ids)),
            "pad_info": pad_info,
            "image_name": pad_info["image_name"]
        }
            
class DynamicPromptAugmentor:
    def __init__(self, concat_prob=0.3, drop_prob=0.1):
        self.concat_prob = concat_prob
        self.drop_prob = drop_prob

    def augment(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        main_prompt = random.choice(prompts)
        
        if len(prompts) > 1 and random.random() < self.concat_prob:
            second_prompt = random.choice(prompts)
            # Avoid duplicating exact string
            if second_prompt != main_prompt:
                if random.random() > 0.5:
                    main_prompt = f"{main_prompt}, {second_prompt}"
                else:
                    main_prompt = f"{second_prompt}, {main_prompt}"
        
        if random.random() < self.drop_prob:
            words = main_prompt.split()
            if len(words) > 3: # Only drop if long enough
                num_drop = random.randint(1, min(3, len(words)//2))
                indices_to_drop = set(random.sample(range(len(words)), num_drop))
                main_prompt = " ".join([w for i, w in enumerate(words) if i not in indices_to_drop])

        return main_prompt

class AugmentedTextSeg(TextSeg):
    def __init__(self, data_dir, text_label_path, image_size=256, 
                 concat_prob=0.3, drop_prob=0.1):
        super().__init__(data_dir, text_label_path, image_size)
        self.augmentor = DynamicPromptAugmentor(concat_prob=concat_prob, drop_prob=drop_prob)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        if not data:
            return data
        dataset_name = data['ds']
        mask = data['mask']
        if dataset_name not in self.text_labels:
            return data 
            
        text_prompt = self.text_labels[dataset_name]        
        class_ids_str = data['class_ids']
        if not class_ids_str: 
            return data
        class_ids = [int(x) for x in class_ids_str.split('&')]
        
        # single target training
        class_id = random.choice(class_ids)
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        mask = (mask == class_id).astype(np.uint8)
        data['mask'] = torch.from_numpy(mask)
        data['class_ids'] = str(class_id)
        raw_prompts = text_prompt[str(class_id)] 
        augmented_prompt = self.augmentor.augment(raw_prompts)
        data['text'] = augmented_prompt
        
        return data


def random_transform(images, masks):
    B, H, W = images.shape
    if torch.rand(1).item() > 0.5:
        images = np.flip(images, axis=2)
        masks = np.flip(masks, axis=2)
        
    if torch.rand(1).item() > 0.5:
        images = np.flip(images, axis=1)
        masks = np.flip(masks, axis=1)
        
    k = random.choice([0, 1, 2, 3]) # 0, 90, 180, 270 degrees
    if k > 0:
        images = np.rot90(images, k=k, axes=(1, 2))
        masks = np.rot90(masks, k=k, axes=(1, 2))
    
    return np.ascontiguousarray(images), np.ascontiguousarray(masks)