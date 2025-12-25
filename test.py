from src.dataset.utils import unpad_and_resize
from src.dataset.textseg import TextSeg, TextSegVal
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
    data_dir='/root/autodl-tmp/dataset/train_sample',
    text_label_path = '/root/autodl-tmp/dataset/seg_class.json',
    n_slicing=8,
    image_size=256,
)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8, )

val_ds = TextSegVal(
    data_dir='/root/autodl-tmp/dataset/train_sample',
    text_label_path = '/root/autodl-tmp/dataset/seg_class.json',
    image_size=256,
)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8,)

opts, schedulers = model.configure_optimizers()
opt = opts[0]
device = torch.device("cuda")
train_batch = next(iter(dl))
for k, v in train_batch.items():
    if isinstance(v, torch.Tensor):
        train_batch[k] = v.to(device)
        
val_batch = next(iter(val_dl))
for k, v in val_batch.items():
    if isinstance(v, torch.Tensor):
        val_batch[k] = v.to(device)

step_id = 0
model.train()

while True:
    opt.zero_grad()
    loss, log_dict, segs = model.training_step(train_batch, step_id)
    loss.backward()
    opt.step()
    
    if step_id % 50 == 0 and step_id != 0:
        print(f"\n=== Iter: {step_id} ===")
        print(f"[Train Loss] Total: {loss.item():.4f} | CLIP : {log_dict.get('train/clip_loss', 0):.4f} | Bg : {log_dict.get('train/bg_loss', 0):.4f} | bce_loss: {log_dict.get('train/bce_loss', 0):.4f} | dice_loss: {log_dict.get('train/dice_loss', 0):.4f}")
        
        model.eval()
        with torch.no_grad():
            try:
                model.validation_step(val_batch, step_id)
                metrics = model.val_metrics.compute()
                print(f"[Val Metrics] DSC: {metrics['dsc']:.4f} | NSD: {metrics['nsd']:.4f} | F1: {metrics['f1']:.4f}")
                model.val_metrics.reset()
            except Exception as e:
                print(f"!!! Validation Pipeline Failed: {e}")
                import traceback
                traceback.print_exc()
        # training log_image
        imgs, gts, texts, class_ids = model.get_input(train_batch)
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
