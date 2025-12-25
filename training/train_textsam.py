import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from src.models.heads import PromptEncoder, MaskDecoder
from src.models.transformer import TwoWayTransformer
import random
from utils.helper import instantiate_from_config
from itertools import chain
from src.dataset.utils import unpad_and_resize
from evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def process_multi_prompts(text):
    """
    Process the input text to handle multiple prompts.
    This function splits the text by [SEP] and returns a list of prompts.
    """
    if text is None:
        return None, None
    text = text if isinstance(text, (list, tuple)) else [text]
    text = [_text.split("[SEP]")for _text in text]    
    num_prompts = torch.tensor([len(_text) for _text in text], dtype=torch.int64)
    text = [t for i in range(len(text)) for t in text[i]]
    return text, num_prompts

def parse_prompt_ids_dense(prompt_ids):
    slice_strings = prompt_ids.split('&')
    all_slices_labels = []
    for content in slice_strings:
        if not content:
            all_slices_labels.append([])
        else:
            all_slices_labels.append([int(x) for x in content.split(':')])
    return all_slices_labels

def compute_multi_class_metrics(gt, seg):
    metrics = {
        ""
    }
    
class TextSAM(pl.LightningModule):
    def __init__(
        self,
        image_encoder,
        text_embedder_configs,
        maskformer_config,
        loss_configs,
        ds_scale = 4.,
        image_size=256,
        hidden_dim=256,
        checkpoint=None,
    ):
        super().__init__()        
        self.image_encoder = instantiate_from_config(image_encoder)
        self.text_embedder = instantiate_from_config(text_embedder_configs)
        self.mask_former = instantiate_from_config(maskformer_config)
        self.loss_fn = instantiate_from_config(loss_configs)
        self.image_size = image_size
        self.image_embedding_size = int(image_size // ds_scale)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self._build_sam_heads()
        if checkpoint is not None:
            load_res = self.load_from_local(checkpoint, strict=False)
            print(load_res)

        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.text_embedder.pooler.parameters():
            p.requires_grad_(True)
        for p in self.mask_former.parameters():
            p.requires_grad_(True)
        for p in self.loss_fn.parameters():
            p.requires_grad_(True)
        
    def load_from_local(self, checkpoint_path, strict=False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                 k = k.replace('model.', '', 1) 
            new_state_dict[k] = v
        load_result = self.load_state_dict(new_state_dict, strict=strict)
        return load_result
    
    def get_input(self, batch):
        images = batch['image']
        masks = batch['mask'].to(torch.long)
        B, N, C, H, W = images.shape
        images = images.reshape(-1, C, H, W).expand(-1, 3, -1, -1)
        B, N, C, H, W = masks.shape
        masks = masks.reshape(-1, C, H, W)
        text = list(chain.from_iterable(x.split("[SEP]") for x in batch['text']))
        class_ids = batch['mask_ids']
        class_ids = class_ids.reshape(-1)
        return images, masks, text, class_ids
    
    def get_val_input(self, batch):
        images = batch['image']
        masks = batch['masks'].to(torch.long)
        if images.shape[1] != 3:
            images = images.expand(-1, 3, -1, -1)
        if masks.shape[1] != 1:
            masks = masks.unsqueeze(1)
        text = list(chain.from_iterable(x.split("[SEP]") for x in batch['text']))
        cls_ids = batch['class_ids'].split("&")
        return images, masks, text, cls_ids
    
    def decode(self, image_embeddings, text_embeddings):
        queries, keys, q_cls, k_cls = self.mask_former(image_embeddings, text_embeddings)
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text=q_cls,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=queries, 
            multimask_output=False,
        )
        return low_res_masks, iou_predictions, k_cls, q_cls
    
    def forward(self, image, text):
        image_embeddings = self.image_encoder(image) # b c h w 
        text_embeddings = self.text_embedder(text) # b c
        if text_embeddings.ndim == 3:
            text_embedding = text_embeddings[:, 0, :]
        text_embedding = text_embedding.unsqueeze(1) # b 1 c
        
        return self.decode(image_embeddings=image_embeddings, text_embeddings=text_embeddings)
    
    
    def training_step(self, batch, batch_idx):
        images, target_masks, text_list, class_ids = self.get_input(batch)
        image_embeddings = self.image_encoder(images) # B*N, d, H, W
        text_embeddings = self.text_embedder(text_list) # B*N*M
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1) # b 1 c
        
        # expand image embeddings
        M = text_embeddings.shape[0] // image_embeddings.shape[0]
        image_embeddings = image_embeddings.unsqueeze(1).expand(-1, M, -1, -1, -1)
        image_embeddings = image_embeddings.reshape(-1, *image_embeddings.shape[2:])

        # decode segs
        low_res_masks, iou_predictions, k_cls, q_cls = self.decode(image_embeddings, text_embeddings)
        
        # loss_inputs
        is_background = (class_ids == 0)
        bg_indices = torch.nonzero(is_background).squeeze(1)
        fg_indices = torch.nonzero(~is_background).squeeze(1)
        img_feat = k_cls.squeeze(1)[fg_indices]
        text_feat = q_cls.squeeze(1)[fg_indices]
        bg_feat = k_cls.squeeze(1)[bg_indices] if len(bg_indices) > 0 else None
        total_loss, log_dict = self.loss_fn(low_res_masks, target_masks, img_feat, text_feat, bg_feat, batch_idx, split='train')
        self.log_dict(log_dict, prog_bar=True, logger=True)
        
        return total_loss, log_dict, low_res_masks
    
    def validation_step(self, batch, batch_idx):
        images, target_masks, text_list, cls_ids = self.get_val_input(batch)
        image_embeddings = self.image_encoder(images) # B*D, d, H, W
        text_embeddings = self.text_embedder(text_list) # BN, C
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1)
        
        all_probs = []
        for id in cls_ids:
            text_embed = text_embeddings[id - 1]
            k_pred_seg, _, _, _ = self.decode(image_embeddings, text_embed) # B,1,H,W
            k_pred_seg = unpad_and_resize(
                k_pred_seg,
                org_size= target_masks.shape[-2:],
                curr_size= k_pred_seg.shape[-1]
            ) 
            all_probs.append(torch.sigmoid(k_pred_seg).squeeze(1).cpu())
        
        all_probs = torch.stack(all_probs, dim=0) # K, D, H, W
        pred_segs = torch.zeros_like(target_masks) 
        max_indices = torch.argmax(all_probs, dim=0)
        max_scores = torch.max(all_probs, dim=0)
        foreground_mask = max_scores > 0.5
        dsc = []
        for k, id in enumerate(cls_ids):
            mask_k = foreground_mask & (max_indices == k)
            pred_segs[mask_k]=id
        
        return total_loss

    def _build_sam_heads(self):
        self.prompt_embed_dim = self.hidden_dim
        self.sam_prompt_embed_dim = self.hidden_dim
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def configure_optimizers(self):
        lr = 1e-5
        new_params = list(self.mask_former.parameters()) + \
                      list(self.text_embedder.pooler.parameters()) + \
                      list(self.loss_fn.parameters())
            
        pretrained_params = list(self.prompt_encoder.parameters()) + \
                            list(self.mask_decoder.parameters())

        pretrained_params = [p for p in pretrained_params if p.requires_grad]
        opt_lr = [
            {
                "params": pretrained_params,
                "lr": lr , 
            },
            {
                "params": new_params,
                "lr": lr ,        
            }
        ]
        optimizer = torch.optim.AdamW(
            opt_lr, 
            betas=(0.9, 0.995),
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        return [optimizer], [scheduler]


    @torch.no_grad()
    def log_images(self, batch, split="train", pl_module=None):
        images, target_masks, text_list = self.get_input(batch)
        pred_masks, iou_pred, q_cls, k_cls = self(images, text_list)
        
        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(target_masks.float(), size=pred_masks.shape[-2:], mode='nearest').long()
            
        pred_masks = torch.sigmoid(pred_masks)
        pred_masks = (pred_masks > 0.5).float()
        
        return {
            "inputs": images,
            "gts": target_masks,
            "outputs": pred_masks,
            "text": text_list
        }
