import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Optional, Type, Tuple
from src.models.heads import PromptEncoder, MaskDecoder, TwoWayTransformer
import random
from utils.helper import instantiate_from_config


def random_transform(images, masks):
    B, C, H, W = images.shape
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[3])
        masks = torch.flip(masks, dims=[3])
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[2])
        masks = torch.flip(masks, dims=[2])
    k = random.choice([0, 1, 2, 3]) # 0, 90, 180, 270 degrees
    if k > 0:
        images = torch.rot90(images, k=k, dims=[2, 3])
        masks = torch.rot90(masks, k=k, dims=[2, 3])
    return images, masks

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

class TextSAM(pl.LightningModule):
    def __init__(
        self,
        image_encoder,
        text_embedder_configs,
        maskformer_config,
        loss_configs,
        ds_scale = 4.,
        image_size=256,
        checkpoint=None,
    ):
        super().__init__()        
        self.image_encoder = image_encoder
        self.text_embedder = instantiate_from_config(text_embedder_configs)
        self.mask_former = instantiate_from_config(maskformer_config)
        self.loss_fn = instantiate_from_config(loss_configs)
        self.image_size = image_size
        self.image_embedding_size = int(image_size // ds_scale)
        self._build_sam_heads()
        self.load_from_local(checkpoint, strict=False)

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
        if len(images.shape) > 4:
            B, C, H, W = images.shape
            images = images.reshape(-1, C, H, W)
        else:
            B, N, H, W = images.shape
            images = images.reshape(-1, H, W).unsqueeze(1).expand(-1, 3, -1, -1)
        if len(masks.shape) > 4:
            B, N, C, H, W = masks.shape
            masks = masks.reshape(-1, C, H, W)
        images, masks = random_transform(images, masks)
        text = batch['text']
        flat_text = [x for t in text for x in t.split('[SEP]')]
        return images, masks, flat_text

    def forward(self, image, text):
        # if any("[SEP]" in t for t in text):
        #     text, num_text = process_multi_prompts(text)
        image_embedding = self.image_encoder(image) # b c h w 
        text_embedding = self.text_embedder(text) # b c
        if len(text_embedding.shape) > 2:
            text_embedding = text_embedding[:, -1]
        text_embedding = text_embedding.unsqueeze(1)
        text_embedding = F.normalize(text_embedding, dim=-1, p=2)
        image_embedding = F.normalize(image_embedding, dim=-1, p=2)
        queries, keys, q_cls, k_cls,  = self.mask_former(image_embedding, text_embedding)
        dense_mask_input = queries
        sparse_embeddings, dense_embeddings = self.prompt_encoder( 
            points=None,
            boxes=None,
            masks=None,
            text=q_cls,
        )
        dense_embeddings = dense_mask_input 
        image_pe = self.prompt_encoder.get_dense_pe() 
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe, 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions, k_cls, q_cls
    
    def training_step(self, batch, batch_idx):
        images, target_masks, text_list = self.get_input(batch)
        pred_masks, iou_pred, k_cls, q_cls = self(images, text_list)
        
        img_feat = k_cls.squeeze(1)
        text_feat = q_cls.squeeze(1)
        
        total_loss, log_dict = self.loss_fn(pred_masks, target_masks, img_feat, text_feat, batch_idx, split='train')
        
        self.log_dict(log_dict, prog_bar=False, logger=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, target_masks, text_list = self.get_val_input(batch)
        pred_masks, _,  q_cls, k_cls = self(images, text_list)
        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(target_masks.float(), size=pred_masks.shape[-2:], mode='nearest').long()
        
        img_feat = k_cls.squeeze(1)
        text_feat = q_cls.squeeze(1)

        total_loss, log_dict = self.loss_fn(pred_masks, target_masks, img_feat, text_feat, batch_idx, split='val')

        self.log_dict(log_dict, prog_bar=False, logger=True)
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
        lr = self.learning_rate
        new_params = list(self.mask_former.parameters()) + \
                      list(self.text_embedder.pooler.parameters()) + \
                      list(self.loss_fn.parameters())
            
        pretrained_params = list(self.image_encoder.parameters()) + \
                            list(self.prompt_encoder.parameters()) + \
                            list(self.mask_decoder.parameters())

        pretrained_params = [p for p in pretrained_params if p.requires_grad]
        opt_lr = [
            {
                "params": pretrained_params,
                "lr": lr , 
            },
            {
                "params": new_params,
                "lr": lr * 0.1,        
            }
        ]
        optimizer = torch.optim.AdamW(
            opt_lr, 
            betas=(0.9, 0.995),
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks =  F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks

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
