import argparse
import os
import sys
import importlib
import torch
import torchvision
import datetime
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
sys.path.append(os.getcwd())

class FabricImageLogger:
    def __init__(self, max_images=4, clamp=True):
        self.max_images = max_images
        self.clamp = clamp

    @torch.no_grad()
    def log_images(self, fabric, model, val_loader, global_step, split="val/"):
        if fabric.global_rank != 0:
            return

        model.eval()
        
        batch = next(iter(val_loader))
        log_dict = model.log_images(batch) 
        
        images = log_dict["inputs"]   # (B, 3, H, W)
        gts = log_dict["gts"]         # (B, 1, H, W)
        outputs = log_dict["outputs"] # (B, 1, H, W)
        prompts = log_dict["text"]
                
        N = min(images.shape[0], self.max_images)
        images = images[:N]
        gts = gts[:N]
        outputs = outputs[:N]
        
        images_uint8 = (images.detach().cpu().float() * 255).clamp(0, 255).to(torch.uint8)

        if gts.ndim == 4: gts = gts.squeeze(1)
        if outputs.ndim == 4: outputs = outputs.squeeze(1)
        
        gts_bool = gts.detach().cpu() > 0.5
        outputs_bool = outputs.detach().cpu() > 0.5
        combined_images = []        
        
        for i in range(N):
            img = images_uint8[i]            
            gt_overlay = draw_segmentation_masks(
                img, 
                masks=gts_bool[i], 
                colors=(0, 255, 0), # Green
                alpha=0.4
            )
            
            pred_overlay = draw_segmentation_masks(
                img, 
                masks=outputs_bool[i], 
                colors=(255, 0, 0), # Red
                alpha=0.4
            )
            
            combined_images.extend([img, gt_overlay, pred_overlay])

        grid = make_grid(combined_images, nrow=3, padding=2)        
        grid = grid.float() / 255.0

        log_dir = fabric.logger.log_dir
        
        if log_dir is None:
            log_dir = "logs" 
            
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            split = split.replace('/', '_')
            save_path = os.path.join(log_dir, f"{split}_{global_step}.png")

            save_image(grid, save_path)

        model.train()

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def instantiate_from_config(config):
    if not (target := config.get("target")):
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", {}))

def get_optimizer_and_scheduler(model, config, steps_per_epoch):
    lr = model.learning_rate
    new_params = list(model.mask_former.parameters()) + \
                  list(model.text_embedder.pooler.parameters())
    if hasattr(model.clip_loss_fn, "parameters"):
        new_params += list(model.clip_loss_fn.parameters())
    
    pretrained_params = list(model.image_encoder.parameters()) + \
                        list(model.prompt_encoder.parameters()) + \
                        list(model.mask_decoder.parameters())

    pretrained_params = [p for p in pretrained_params if p.requires_grad]
    
    opt_lr = [
        {"params": pretrained_params, "lr": lr * 0.1},
        {"params": new_params, "lr": lr * 0.05}
    ]
    
    optimizer = torch.optim.AdamW(opt_lr, betas=(0.9, 0.995), weight_decay=0.05)
    
    # 修复 scheduler：使用总步数
    total_steps = config.lightning.trainer.max_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    return optimizer, scheduler

def train_one_epoch(fabric, model, train_loader, optimizer, scheduler, epoch, global_step):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        images, target_masks, text_list = model.get_input(batch)
        pred_masks, iou_pred, k_cls, q_cls = model(images, text_list)
        
        loss_seg, log_dict = model.loss_fn(pred_masks, target_masks, split='train')
        img_feat = k_cls.squeeze(1)
        text_feat = q_cls.squeeze(1)
        clip_loss = model.clip_loss_fn(img_feat, text_feat, batch_idx)
        
        total_loss = loss_seg + model.clip_loss_weight * clip_loss
        
        fabric.backward(total_loss)
        optimizer.step()
        scheduler.step()
        
        fabric.log("train/total_loss", total_loss, step=global_step)
        fabric.log("train/seg_loss", loss_seg, step=global_step)
        fabric.log("train/clip_loss", clip_loss, step=global_step)
        
        if batch_idx % 50 == 0:
            fabric.print(f"Ep {epoch} | Step {global_step} | Loss: {total_loss.item():.4f}")
            
        global_step += 1
    
    return global_step

def validate_one_epoch(fabric, model, val_loader, epoch, global_step):
    model.eval()
    avg_metrics = {}
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images, target_masks, text_list = model.get_input(batch)
            pred_masks, _, q_cls, k_cls = model(images, text_list)
            
            if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                target_masks = torch.nn.functional.interpolate(
                    target_masks.float(), size=pred_masks.shape[-2:], mode='nearest'
                ).long()
            
            loss_seg, log_dict = model.loss_fn(pred_masks, target_masks, split='val')
            clip_loss = model.clip_loss_fn(k_cls.squeeze(1), q_cls.squeeze(1))
            total_loss = loss_seg + model.clip_loss_weight * clip_loss
            
            # DDP Sync
            total_loss_synced = fabric.all_gather(total_loss).mean()
            clip_loss_synced = fabric.all_gather(clip_loss).mean()
            
            if "val/total_loss" not in avg_metrics: avg_metrics["val/total_loss"] = 0.0
            avg_metrics["val/total_loss"] += total_loss_synced.item()
            
            if "val/clip_loss" not in avg_metrics: avg_metrics["val/clip_loss"] = 0.0
            avg_metrics["val/clip_loss"] += clip_loss_synced.item()
            
            for k, v in log_dict.items():
                if k not in avg_metrics: avg_metrics[k] = 0.0
                v_tensor = v if isinstance(v, torch.Tensor) else torch.tensor(v, device=fabric.device)
                v_synced = fabric.all_gather(v_tensor).mean()
                avg_metrics[k] += v_synced.item()

    fabric.print(f"\n--- Validation Epoch {epoch} Summary ---")
    for k, v in avg_metrics.items():
        avg_value = v / num_batches
        fabric.log(k, avg_value, step=global_step)
        if "dice" in k or "loss" in k:
            fabric.print(f"{k}: {avg_value:.4f}")
            
    return avg_metrics.get("val/total_loss", 0.0)

# ==============================================================================
# 3. Main Logic
# ==============================================================================
def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", nargs="*", default="configs/text_seg_repvit.yaml")
    parser.add_argument("-s", "--seed", type=int, default=1234)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("-n", "--name", type=str, default="", help="10percent_5_fold")
    args = parser.parse_args()
    
    logger = None
    configs = [OmegaConf.load(c) for c in args.base]
    config = OmegaConf.merge(*configs)
    
    cfg_name = os.path.splitext(os.path.basename(args.base[0]))[0] if args.base else "run"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_name = f"{timestamp}_{cfg_name}"
    if args.name: experiment_name += f"_{args.name}"
    
    print(f"Logs will be saved to: logs/{experiment_name}")

    train_dataset_cfg = config.data.params.train
    full_dataset = instantiate_from_config(train_dataset_cfg)
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    image_logger = FabricImageLogger(max_images=4)
    
    batch_size = config.data.params.batch_size
    num_workers = config.data.params.num_workers
    max_epochs = config.lightning.trainer.max_epochs
    batch_freq = config.lightning.callbacks.image_logger.params.batch_frequency
    save_every_n_step = config.trainer.save_every_n_step
    # K-Fold Loop
    for fold, (train_ids, val_ids) in enumerate(kf.split(full_dataset)):
        print(f"\n{'='*20} Starting Fold {fold+1}/{args.folds} {'='*20}")
        
        # --- 初始化 Best Loss ---
        # 每个 Fold 开始前，重置最佳 Loss 为无穷大
        best_val_loss = float('inf')

        strategy = "ddp_find_unused_parameters_true" if str(args.devices) != "1" else "auto"
        
        fabric = Fabric(
            accelerator="gpu", 
            devices=args.devices, 
            precision="16-mixed",
            strategy=strategy
        )
        fabric.launch()
        seed_everything(args.seed)
        ckpt_dir = os.path.join("logs", experiment_name, f"fold_{fold}", "checkpoints")

        if fabric.global_rank == 0:
            logger = TensorBoardLogger(root_dir="logs", name=experiment_name, version=f"fold_{fold}")
            fabric._loggers = [logger]
            os.makedirs(ckpt_dir, exist_ok=True)
        fabric.barrier()
        
        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
        
        # 3. Model & Optim
        model = instantiate_from_config(config.model)
        
        num_devices = fabric.world_size
        model.learning_rate = 1 * num_devices * batch_size * config.model.base_learning_rate
        
        optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))
        model, optimizer = fabric.setup(model, optimizer)
        model.mark_forward_method('log_images')
        # 4. Training Loop
        global_step = 0
        try:
            for epoch in range(1, max_epochs + 1):
                global_step = train_one_epoch(fabric, model, train_loader, optimizer, scheduler, epoch, global_step)
                
                # 获取当前的 validation loss
                val_loss = validate_one_epoch(fabric, model, val_loader, epoch, global_step)
                
                # --- Image Logger ---
                if global_step % batch_freq == 0 or global_step == 1:
                    image_logger.log_images(fabric, model, val_loader, global_step, tag=f"val_fold{fold}/")
                
                # --- Checkpointing Logic ---
                
                state = {
                    "model": model, 
                    "optimizer": optimizer, 
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss
                }
                save_path_last = os.path.join(ckpt_dir, "last.ckpt")
                fabric.save(save_path_last, state)
                
                if val_loss < best_val_loss:
                    old_best = best_val_loss
                    best_val_loss = val_loss
                    
                    save_path_best = os.path.join(ckpt_dir, "best.ckpt")
                    fabric.save(save_path_best, state)
                    
                    if fabric.global_rank == 0:
                        fabric.print(f"🔥 New best model found! Loss improved {old_best:.4f} -> {best_val_loss:.4f}. Saved to {save_path_best}")

                if global_step % save_every_n_step == 0:
                    save_path_epoch = os.path.join(ckpt_dir, f"{epoch:04d}-{global_step:06d}.ckpt")
                    fabric.save(save_path_epoch, state)
                    
        except KeyboardInterrupt:
                if fabric.global_rank == 0:
                    print(f"\n[Warning] Training interrupted by user at Epoch {epoch}, Step {global_step}!")
                    print("Saving emergency checkpoint to 'last.ckpt'...")
                state = {
                    "model": model, 
                    "optimizer": optimizer, 
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss if 'val_loss' in locals() else float('inf')
                }
                save_path_last = os.path.join(ckpt_dir, "last.ckpt")
                fabric.save(save_path_last, state)
                sys.exit(0)

        if fabric.global_rank == 0:
            print(f"Fold {fold+1} finished successfully.")

if __name__ == "__main__":
    main()