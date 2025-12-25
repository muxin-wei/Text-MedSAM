from src.dataset.utils import unpad_and_resize
from src.dataset.textseg import TextSeg
from torch.utils.data import DataLoader
from utils.helper import instantiate_from_config
from omegaconf import OmegaConf
from itertools import chain
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os
import numpy as np
import random

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
# np.random.seed()
config_path = "configs/text_seg_repvit.yaml"
config = OmegaConf.load(config_path)

model = instantiate_from_config(config.model).to("cuda")


ds = TextSeg(
    data_dir='/root/autodl-tmp/dataset/train_10/CT',
    text_label_path = '/root/autodl-tmp/dataset/seg_class.json',
    n_slicing=8,
    image_size=256,
)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=12, )


opts, schedulers = model.configure_optimizers()
opt = opts[0]
device = torch.device("cuda")
batch = next(iter(dl))


step_id = 0
model.train()
while True:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    opt.zero_grad()
    loss, log_dict, segs = model.training_step(batch, step_id)
    loss.backward()
    opt.step()
    
    if step_id % 50 == 0:
        print(f"iter:{step_id}, loss: {loss.item()}, clip_loss: {log_dict["train/clip_loss"]}, bg_loss: {log_dict["train/bg_loss"]}, bce_loss: {log_dict["train/bce_loss"]}, dice_loss: {log_dict["train/dice_loss"]}")
        model.eval()
        imgs, gts, texts, class_ids = model.get_input(batch)
        M = gts.shape[0] // imgs.shape[0]
        is_background = (class_ids == 0)
        fg_indices = torch.nonzero(~is_background).squeeze(1)
        idx = random.choice(fg_indices)
        
        img = imgs[idx // M ].permute(1, 2, 0).cpu().numpy()
        gt = gts[idx].cpu().permute(1, 2, 0).cpu().numpy()
        seg = torch.sigmoid(segs[idx]).detach().cpu()
        seg = (seg > 0.5).numpy().squeeze()
        text = texts[idx]
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img)
        ax[0].axis('off')
        
        ax[1].imshow(img)
        ax[1].imshow(seg, cmap='Reds', vmin=0, vmax=1, alpha=0.5)
        ax[1].axis('off')
        
        ax[2].imshow(img)
        ax[2].imshow(gt, cmap='Greens', alpha=0.5)
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(f"./test/vis_{step_id:06d}.png")
        plt.close()
    step_id+=1
    model.train()
