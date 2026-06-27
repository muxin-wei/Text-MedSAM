from src.losses.medsam_loss import DiceLoss
from torch import nn
from torch.nn import functional as F
import torch
import math

class TextSegLoss(nn.Module):
    def __init__(self, 
                 dice_weight=1.0,
                 focal_weight=0.5,
                 contrastive_weight=0.3,
                #  boundary_weight=0.2,
                #  dice_threshold=0.6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        # self.boundary_weight = boundary_weight
        # self.dice_threshold = dice_threshold
        
        self.dice = DiceLoss(sigmoid=True, to_onehot_y=False, reduction='mean')
        self.focal = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
        # self.boundary = BoundaryLoss()
        self.contrastive = ContrastiveLoss(temperature=0.07)  
   
    def forward(self, logits, target, batch_idx, text_embed=None, img_feat=None, split='train'):
        dice_loss = self.dice_weight * self.dice(logits, target)
        focal_loss = self.focal_weight * self.focal(logits, target)
        
        # current_dice = 1.0 - dice_loss.item()
        # boundary_w = self.boundary_weight * max(0.0, (self.boundary_start_dice - current_dice) / 0.3)
        
        # if boundary_w > 0:
            # boundary_loss = boundary_w * self.boundary(logits, target)
        contrastive_loss = torch.tensor(0.0, device=logits.device)
        if self.contrastive_weight > 0 and text_embed is not None and img_feat is not None:
            contrastive_loss = self.contrastive_weight * self.contrastive(img_feat, text_embed, target)
            
        loss = dice_loss + focal_loss + contrastive_loss 
            # +  boundary_loss\
        log_dict = {
            f'{split}/loss': loss.detach(),
            f"{split}/dice_loss": dice_loss.detach(),
            f"{split}/focal_loss": focal_loss.detach(),
            f"{split}/contrastive_loss": contrastive_loss.detach(),
            # f"{split}/boundary_loss": boundary_loss.detach() if boundary_w > 0 else torch.nan
        }
        
        return loss, log_dict

class FocalLoss(nn.Module):
    def __init__(self, gamma: float=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs) # prob_map
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(temperature)))
    
    def forward(self, img_feats: torch.Tensor, text_embed: torch.Tensor, target: torch.Tensor):
        B, C, H, W = img_feats.shape
        img_feats = img_feats.flatten(2) # 
        target = target.clone().flatten(2) # [B, 1, N]
        fg_mask = (target > 0.5).float().view(B, 1, -1)
        bg_mask = (target <= 0.5).float().view(B, 1, -1)
        
        fg_count = fg_mask.sum(dim=-1).squeeze(-1)
        bg_count = bg_mask.sum(dim=-1).squeeze(-1)
        is_valid = (fg_count > 10) & (bg_count > 10)
        if not is_valid.any():
            return torch.tensor(0.0, device=img_feats.device, requires_grad=True)
        
        img_valid = img_feats[is_valid]       # [K, C, N]
        fg_valid = fg_mask[is_valid]         # [K, 1, N]
        bg_valid = bg_mask[is_valid]         # [K, 1, N]
        text_embed = text_embed[is_valid]    # [K, 1, C]
        
        fg_proto = (img_valid * fg_valid).sum(dim=-1) / (fg_valid.sum(dim=-1) + 1e-6)
        fg_proto = F.normalize(fg_proto, dim=-1) # [K, C]
        bg_proto = (img_valid * bg_valid).sum(dim=-1) / (bg_valid.sum(dim=-1) + 1e-6)
        bg_proto = F.normalize(bg_proto, dim=-1) 

        text_embed = F.normalize(text_embed.squeeze(1), dim=-1)
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))
        scale = self.logit_scale.exp().clamp(max=100.)
        pos_sim = (fg_proto * text_embed).sum(dim=-1, keepdim=True) * scale
        neg_sim = (bg_proto * text_embed).sum(dim=-1, keepdim=True) * scale
        logits = torch.cat([pos_sim, neg_sim], dim=1) # [K, 2]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction="mean") 
        return loss
    
