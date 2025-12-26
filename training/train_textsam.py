import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from src.models.heads import PromptEncoder, MaskDecoder
from src.models.transformer import TwoWayTransformer
from utils.helper import instantiate_from_config
from itertools import chain
from src.dataset.utils import unpad_and_resize
from src.dataset.eval import SegEval
from time import time

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
            
        # new params (pooler, mask_former, logit_scale)
        for p in self.text_embedder.pooler.parameters():
            p.requires_grad_(True)
        for p in self.mask_former.parameters():
            p.requires_grad_(True)
        for p in self.loss_fn.parameters():
            p.requires_grad_(True)
        
        self.val_metrics = SegEval()
        
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
        images = batch['image'].squeeze()
        masks = batch['mask'].squeeze()
        if images.shape[1] != 3:
            images = images.unsqueeze(1).expand(-1, 3, -1, -1)

        text = list(chain.from_iterable(x.split("[SEP]") for x in batch['text']))
        is_instance = batch['is_instance'][0] if isinstance(batch['is_instance'], list) else batch['is_instance']
        cls_ids = list(int(x) for x in batch['class_ids'][0].split("&"))
        return images, masks, text, cls_ids, is_instance
    
    @torch.no_grad()
    def encode_image(self, images):
        image_embeddings = self.image_encoder(images)
        return image_embeddings
    
    # def encode_text(self, texts):
    #     with torch.no_grad():
    #         text_embeddings = self.text_embedder.hf_module(texts)
    #     return self.text_embedder.pooler(text_embeddings)
    
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
        image_embeddings = self.encode_image(images) # B*N, d, H, W
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
        images, gt_mask, text_list, class_ids, is_instance = self.get_val_input(batch)
        t0 = time()
        image_embeddings = self.image_encoder(images) 
        text_embeddings = self.text_embedder(text_list) # BN, C
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1)
        
        all_probs = []
        for idx, cls_id in enumerate(class_ids):
            text_embed = text_embeddings[idx].unsqueeze(1)
            k_pred_seg, _, _, _ = self.decode(image_embeddings, text_embed) # D,1,H,W
            k_pred_seg = unpad_and_resize(
                k_pred_seg,
                org_size= gt_mask.shape[-2:],
                curr_size= k_pred_seg.shape[-1]
            ) 
            all_probs.append(torch.sigmoid(k_pred_seg).squeeze(1))
        all_probs = torch.stack(all_probs, dim=0)
        t1 = time()
        D, H, W = all_probs.shape[1:]
        pred_segs = torch.zeros((D, H, W), dtype=torch.uint8, device=all_probs.device) # D,H,W
        if is_instance:
            pred_segs = (all_probs[0] > 0.5)
        else:
            max_scores, max_indices = torch.max(all_probs, dim=0)
            fg_masks = max_scores > 0.5 
            for k, cls_id in enumerate(class_ids):
                mask_k = fg_masks & (max_indices == k) # extraction for k-th seg
                pred_segs[mask_k] = int(cls_id)
        pred_segs = pred_segs.long()
        t2 = time()
        if gt_mask.ndim == 5: gt_mask = gt_mask.squeeze()
        self.val_metrics.update(pred_segs, gt_mask, class_ids)
        print(f"\r[Val Step {batch_idx}] Infer: {t1-t0:.2f}s | Metric: {t2-t1:.2f}s", end="")        
        
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)
        self.val_metrics.reset()
        
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
        params = list(self.mask_former.parameters()) + \
                      list(self.text_embedder.pooler.parameters()) + \
                      list(self.loss_fn.parameters())
        opt_lr = [
            {
                "params": params,
                "lr": lr , 
            },
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
