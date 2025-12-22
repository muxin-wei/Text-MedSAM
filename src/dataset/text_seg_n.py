import os
import os.path as osp
import glob
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils import process_input, get_axis, choose_prompt, process_output

class TextSeg(Dataset):
    def __init__(self, data_dir, text_label_path, image_size=256, interpolate_mask_size=256, n_slicing=3):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g., dataset/train_10/)
            text_label_path (str): Path to the text label JSON file.
            image_size (int): Target image size for resizing (default: 256).
            mode (str): 'train' or 'val'. In train mode, tries to pick slices with labels.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.interpolate_mask_size = interpolate_mask_size
        self.n_slicing = n_slicing
        with open(text_label_path, 'r') as f:
            self.text_labels = json.load(f)
        valid_dataset = set(self.text_labels.keys())
        self.samples = glob.glob(osp.join(data_dir, '**/*.npz'),recursive=True)
        self.samples = sorted([
            file_path 
            for file_path in self.samples 
            if osp.basename(osp.dirname(file_path)) in valid_dataset
        ])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        dataset_name = osp.basename(osp.dirname(file_path))
        try:
            data = np.load(file_path)
            imgs = data['imgs'].astype(np.uint8)
            mask = data['gts'].astype(np.uint8)
            text_prompt = self.text_labels[dataset_name]
        except Exception as e:
            print(f'error loading {file_path}: {e}')
            return None
        D, H, W = imgs.shape

        # if D < self.n_slicing:
        #     num_pad = self.n_slicing - D
        #     indices = np.concatenate([np.arange(D), np.full(num_pad, D - 1)]).astype(int)
        # else:
        #     indices = np.random.choice(np.arange(D), size=self.n_slicing)
        images = imgs
        # images = imgs[indices]
        # mask = mask[indices]
        image, pad_width, padded_size, valid_axis = process_input(images, self.image_size, mode='bicubic')
        prompt_labels = {int(k) for k in text_prompt.keys() if k.isdigit()}
        instance_label = text_prompt.get('instance_label')
        is_binary_mode = (instance_label > 0)
        if is_binary_mode > 0: # binary segmentation
            if instance_label in prompt_labels:
                class_ids = [instance_label]*self.n_slicing
            else:
                class_ids = []
            if class_ids:
                new_mask = (mask == instance_label).astype(np.uint8)
            prompts = random.choices(text_prompt[str(instance_label)], k=self.n_slicing)
        else:
            class_ids = []
            prompts = []
            new_mask = []
            for i in range(mask.shape[0]):
                class_ids_slice = set(np.unique(mask[i]))
                class_ids_slice = [l for l in class_ids_slice if l > 0 and l in prompt_labels]
                if class_ids_slice:
                    cls_id = random.choice(class_ids_slice)
                    new_mask.append((mask[i] == cls_id).astype(np.uint8))
                    class_ids.append(cls_id)
                    prompts.append(random.choice(text_prompt[str(cls_id)]))
                else:
                    class_ids.append(0)
                    prompts.append("background")
                    new_mask.append(np.zeros_like(mask[i]))
                    
        mask = np.stack(new_mask) if isinstance(new_mask, list) else new_mask
        class_ids = "&".join([str(cls) for cls in class_ids]) 
        text_prompt_sep = "[SEP]".join(prompts)
        img_tensor = torch.from_numpy(image).float() / 255.0
        return{
            "image": img_tensor,
            "mask": torch.from_numpy(mask[:,np.newaxis, ...]).to(torch.uint8),
            "class_ids": class_ids,
            "text": text_prompt_sep,
            "ds": dataset_name,
            "img_name": osp.basename(file_path).split('.npz')[0]
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
    def __init__(self, data_dir, text_label_path, image_size=256, interpolate_mask_size=256, 
                 concat_prob=0.3, drop_prob=0.1):
        super().__init__(data_dir, text_label_path, image_size, interpolate_mask_size)
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

class ValTextSeg(Dataset):
    def __init__(self, img_dir, gt_dir, image_size=256):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.img_samples = sorted(glob.glob(osp.join(img_dir, '*.npz'), recursive=True))
        self.data_pairs=[]
        for img_path in self.img_samples:
            basename = osp.basename(img_path)
            gt_path = osp.join(gt_dir, basename)
            if osp.exists(gt_path):
                self.data_pairs.append((img_path, gt_path))
            else:
                print(f"[Warning] GT not found for {basename}, skipping.")
                
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.data_pairs[idx]
        file_name = osp.basename(img_path)
        try:
            val_img_data = np.load(img_path, allow_pickle=True)
            val_gt_data = np.load(gt_path, allow_pickle=True)

            imgs = val_img_data['imgs'].astype(np.uint8)          # (D, H, W)
            masks = val_gt_data['gts'].astype(np.uint8)           # (D, H, W)
            text_prompt_dict = val_img_data['text_prompts'].item() 
            
            imgs, masks , pad_width, padded_size, valid_axis = process_input(imgs, masks, self.image_size)
            prompt_labels = {int(k) for k in text_prompt_dict.keys() if k.isdigit()}
            instance_label = int(text_prompt_dict.get('instance_label', 0))
            is_binary_mode = (instance_label > 0) and (instance_label in prompt_labels)
            valid_indices = []
            D, H, W = masks.shape
            for i in range(D):
                unique_labels = np.unique(masks[i])
                valid_labels = [str(l) for l in unique_labels if l > 0 and l in prompt_labels]
                valid_indices.append(':'.join(valid_labels))
            prompt_ids = '&'.join(valid_indices)
            img_tensor = torch.from_numpy(imgs).float() / 255.0
            return{
                "image": img_tensor,
                "mask": torch.from_numpy(masks[:,np.newaxis, ...]).to(torch.uint8),
                "promt_dict": text_prompt_dict,
                "prompt_ids": prompt_ids,
            }
        except Exception as e:
            print(f'Error loading {file_name}: {e}')
            return self.__getitem__((idx+1)%len(self))