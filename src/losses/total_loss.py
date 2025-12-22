import torch
import torch.nn as nn
from utils.helper import instantiate_from_config

class TotalLoss(nn.Module):
    def __init__(self, seg_loss_configs, clip_loss_configs, clip_loss_weight=1.0):
        super().__init__()
        self.seg_loss = instantiate_from_config(seg_loss_configs)
        self.clip_loss = instantiate_from_config(clip_loss_configs)
        self.clip_loss_weight = clip_loss_weight

    def forward(self, pred_masks, target_masks, img_feat, text_feat, batch_idx, split='train'):
        # Calculate segmentation loss
        loss_seg, log_dict_seg = self.seg_loss(pred_masks, target_masks, split=split)

        # Calculate CLIP loss
        clip_loss = self.clip_loss(img_feat, text_feat, batch_idx)
        
        # Combine losses
        total_loss = loss_seg + self.clip_loss_weight * clip_loss

        # Prepare log dictionary
        log_dict = {f'{split}/total_loss': total_loss.detach()}
        log_dict.update(log_dict_seg)
        log_dict[f'{split}/clip_loss'] = clip_loss.detach()

        return total_loss, log_dict
