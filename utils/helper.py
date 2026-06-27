import os
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
import wandb
from PIL import Image
import numpy as np
import torch
import collections
from itertools import repeat
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from typing import Tuple
import importlib

def instantiate_from_config(config):
    if not (target := config.get("target")):
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", {}))


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)

def create_overlay(image: torch.Tensor, mask: torch.Tensor, color: Tuple[int, int, int] = (255, 0, 0)) -> torch.Tensor:
    """
    Creates an overlay of a binary mask onto an image.
    Args:
        image: Original image tensor (3, H, W) in [0, 1] or [-1, 1].
        mask: Binary mask tensor (1, H, W) in {0, 1}.
        color: RGB color tuple for the mask.
    Returns:
        Overlay image tensor (3, H, W).
    """
    if image.min() < 0:
        image = (image + 1.0) / 2.0 
        
    mask = mask.squeeze(0).bool() 
    color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(3, 1, 1) / 255.0
    mask_colored = color_tensor * mask
    
    alpha = 0.5 
    overlay = image.clone()
    for c in range(3):
        overlay[c, mask] = (1 - alpha) * image[c, mask] + alpha * mask_colored[c, mask]
    
    return overlay

def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


if __name__ == "__main__":
    config = {"keya": "a",
              "keyb": "b",
              "keyc":
                  {"cc1": 1,
                   "cc2": 2,
                   }
              }
    from omegaconf import OmegaConf
    config = OmegaConf.create(config)
    print(config)
    retrieve(config, "keya")

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
    
    @rank_zero_only 
    def on_train_start(self, trainer, pl_module):
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
            
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl_loggers.WandbLogger: self._wandb,
            pl_loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        data = [] 
        texts = images.pop("text", None)
        
        for k in images:
            grid = make_grid(images[k], nrow=4)
            grids[f"{split}/{k}"] = wandb.Image(grid)
            
            if k == 'Overlay_Pred' and texts is not None:
                N = min(images[k].shape[0], len(texts))
                for i in range(N):
                    # Convert tensor to PIL image for WandB
                    img_tensor = images[k][i] # 3, H, W
                    if img_tensor.min() < 0:
                         img_tensor = (img_tensor + 1.0) / 2.0
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    data.append([
                        wandb.Image(Image.fromarray(img_np)), 
                        texts[i]
                    ])
        pl_module.logger.experiment.log(grids)
        
        if data:
            columns = [f"{split}/Overlay_Pred", f"{split}/Prompt"]
            table = wandb.Table(data=data, columns=columns)
            pl_module.logger.experiment.log({f"{split}/Segmentation_Results": table})

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        text_data = images.pop("text", None)
        for k in images:
            grid = make_grid(images[k], nrow=4)
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)
        if text_data:
             pl_module.logger.experiment.add_text(f"{split}/text", str(text_data), global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        
        text_data = images.pop("text", None)
        if text_data:
            filename_txt = "text_gs-{:06}_e-{:06}_b-{:06}.txt".format(
                global_step,
                current_epoch,
                batch_idx)
            path_txt = os.path.join(root, filename_txt)
            with open(path_txt, 'w') as f:
                f.write(str(text_data))

        for k in images:
            grid = make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            Image.fromarray(grid).save(path)
    
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            text_data = images.pop("text", None)
            
            # Create overlays
            inputs = images.get("inputs", None)
            gts = images.get("gts", None)
            outputs = images.get("outputs", None)
            
            if inputs is not None:
                # Assuming inputs are [B, 3, H, W]
                # gts and outputs are [B, 1, H, W]
                
                N = min(inputs.shape[0], self.max_images)
                
                if gts is not None:
                    overlay_gts = []
                    for i in range(N):
                        overlay_gts.append(create_overlay(inputs[i].detach().cpu(), gts[i].detach().cpu(), color=(0, 255, 0))) # Green for GT
                    images["Overlay_GT"] = torch.stack(overlay_gts)
                    
                if outputs is not None:
                    overlay_preds = []
                    for i in range(N):
                        overlay_preds.append(create_overlay(inputs[i].detach().cpu(), outputs[i].detach().cpu(), color=(255, 0, 0))) # Red for Pred
                    images["Overlay_Pred"] = torch.stack(overlay_preds)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            
            if text_data is not None:
                images["text"] = text_data

            self.log_local(pl_module.logger.save_dir, split, images.copy(), # Pass copy to preserve text for next logger
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
def to_ntuple(n, x):
    return _ntuple(n)(x)
