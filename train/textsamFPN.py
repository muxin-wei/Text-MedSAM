import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from utils.helper import instantiate_from_config
from itertools import chain
from transformers import BertConfig, BertModel, BertTokenizer

class TextEmbeddingBank(nn.Module):
    def __init__(self, pt_path, out_dim: int=256):
        super().__init__()
        info = torch.load(pt_path, map_location='cpu')
        in_dim = info["embeddings"].shape[-1]
        hidden_size = int((in_dim + out_dim) / 2)
        self.tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/Text-MedSAM/ckpts/biomedclip", local_files_only=True)
        self.model = BertModel.from_pretrained("/root/autodl-tmp/Text-MedSAM/ckpts/biomedbert")
        self.model.load_state_dict(torch.load("/root/autodl-tmp/Text-MedSAM/ckpts/bert.pt"), strict=False)
        self.model.eval()
        
        self.register_buffer("embeddings", info["embeddings"])
        self.register_buffer('valid_counts', info['valid_text_counts'])
        self.pooler = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim)
        )
        
    def forward(self,text=None, ds_ids=None, c_ids=None, mode="train"):
        if mode == "train":
            return self.forward_train(ds_ids, c_ids)
        else:
            token_ids = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ).to(self.model.device)
            text_features = self.model(
                input_ids=token_ids["input_ids"],
                attention_mask=token_ids['attention_mask']
            )["last_hidden_state"]
            batch_embeds = self.pooler(text_features)
        return batch_embeds
    
    def forward_train(self, ds_ids, c_ids):
        counts = self.valid_counts[ds_ids, c_ids]
        random_txt_ids = (torch.rand(counts.shape, device=counts.device) * counts).long()
        batch_embeds = self.embeddings[ds_ids, c_ids, random_txt_ids]
        batch_embeds = self.pooler(batch_embeds)
        return batch_embeds

class TextSAM(pl.LightningModule):
    def __init__(
        self,
        image_encoder,
        text_embed_mapper,
        maskformer_config,
        maskdecoder_config,
        loss_configs,
        ds_scale = 4.,
        image_size=256,
        hidden_dim=256,
        checkpoint=None,
    ):
        super().__init__()        
        self.image_encoder = instantiate_from_config(image_encoder)
        self.text_embedder = instantiate_from_config(text_embed_mapper)
        self.mask_former = instantiate_from_config(maskformer_config)
        self.mask_decoder = instantiate_from_config(maskdecoder_config)
        self.loss_fn = instantiate_from_config(loss_configs)
        self.image_size = image_size
        self.image_embedding_size = int(image_size // ds_scale)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        if checkpoint is not None:
            self.load_from_local(checkpoint, strict=False)


    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            for p in self.parameters():
                p.requires_grad = False
            
            trainable_modules = [
                self.text_embedder.pooler,
                self.mask_former,
                self.mask_decoder,
                # self.loss_fn,
            ]
            for m in trainable_modules:
                for p in m.parameters():
                    p.requires_grad = True
            
            if hasattr(self.mask_former, 'text_ema'):
                ema = self.mask_former.text_ema
                if isinstance(ema, nn.Parameter):
                    ema.requires_grad = False
                else:
                    for p in ema.parameters():
                        p.requires_grad = False        

            if not hasattr(self, "_printed_trainable"):
                trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.parameters())
                print(f"🔥 Trainable: {trainable:,} / {total:,} "
                    f"({trainable/total*100:.2f}%)")
                self._printed_trainable = True
                    
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
        self.load_state_dict(new_state_dict, strict=strict)
    
    def get_input(self, batch):
        images = batch['image']
        masks = batch['mask'].to(torch.float)
        ds_ids = batch['ds_id'].tolist()
        c_ids = batch['cls_id'].tolist()
        images = images.expand(-1, 3, -1, -1)
        return images, masks, ds_ids, c_ids
    
    def get_val_input(self, batch):
        images = batch["image"]           # (D, 3, 256, 256)
        masks  = batch["mask"]            # (D, 1, 256, 256) semantic mask
        
        all_prompts_raw = batch["all_prompts"] 
        all_prompts = [p.split(" [SEP] ") for p in all_prompts_raw]
        
        # 🌟 Reversing Class IDs: String -> List of Lists of Ints
        ids_raw = batch["prompt_class_ids"]
        prompt_class_ids = [[int(i) for i in ids.split(",") if i] for ids in ids_raw]
        
        pad_info = batch["pad_info"]
                
        return images, masks, prompt_class_ids, all_prompts, pad_info
    
    @torch.no_grad()
    def encode_image(self, images):
        embeddings, out = self.image_encoder(images)
        return out, embeddings
    
    @torch.no_grad()
    def encode_text(self, text):
        return self.text_embedder(text, mode="")
    
    def forward(self, image, text):
        img_embed, feats = self.encode_image(image) 
        text_embeddings = self.text_embedder(text, mode="") 
        
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1)
        
        output, text_out, attn_weights = self.mask_former(img_embed, text_embeddings, feats)
        logits = self.mask_decoder(output, text_embeddings)
        return logits, attn_weights, feats, 
    
    def training_step(self, batch, batch_idx):
        images, masks, ds_ids, c_ids = self.get_input(batch)
        with torch.no_grad():
            img_embed, feats = self.encode_image(images) # B*N, d, H, W
 
        text_embeddings = self.text_embedder(ds_ids, c_ids) # B*N*M
        
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1)
        feats = feats[-2::-1]
        output, text, attn_weights = self.mask_former(img_embed, text_embeddings, feats)
        # output : b, d, h, w
        # text: b, 1, d
        logits = self.mask_decoder(output, text_embeddings)
        
        total_loss, log_dict = self.loss_fn(logits, masks, batch_idx, text_embed=text, img_feat=output, split='train')
        for k, v in log_dict.items():
            self.log(f"{k}", v, sync_dist=True)
        return total_loss
    
    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     images, gt_masks, prompt_class_ids, all_prompts, pad_info = self.get_val_input(batch)
    #     img_embed, feats = self.encode_image(images.squeeze(1))
    #     flattened_prompts = [p for sublist in all_prompts for p in sublist]
    #     text_embeddings = self.text_embedder(text=flattened_prompts, ds_ids=None, c_ids=None, mode="val")
    #     if text_embeddings.ndim == 3:
    #         text_embeddings = text_embeddings[:, 0, :]
    #     text_embeddings = text_embeddings.unsqueeze(1)  # (K, 1, d)
        
    #     feats = feats[-2::-1]
    #     self.print(text_embeddings.shape, img_embed.shape)
    #     output, text, attn_weights = self.mask_former(img_embed, text_embeddings, feats)
    #     logits = self.mask_decoder(output, text)   # (K, 1, H, W)
        
    #     pred_masks = (torch.sigmoid(logits) > 0.5).float()  # (K, 1, H, W)
    #     total_val_dice = 0.0
        
    #     num_prompts = len(flattened_prompts)
    #     for i in range(num_prompts):
    #         pred_i = pred_masks[i]
            
    #         target_class = prompt_class_ids[i]
    #         gt_i = (gt_masks == target_class).float()
    #         inter = (pred_i * gt_i).sum()
    #         union = pred_i.sum() + gt_i.sum()
    #         dice = (2.0 * inter + 1e-5) / (union + 1e-5)
    #         total_val_dice += dice
    #     avg_dice = total_val_dice / num_prompts
    #     total_loss, log_dict = self.loss_fn(logits, gt_masks, batch_idx, 
    #                                         text_embed=text, img_feat=output, split='val')
        
    #     self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
    #     self.log("val/dice_score", avg_dice, prog_bar=True, sync_dist=True) 
        
    #     for k, v in log_dict.items():
    #         self.log(f"val/{k}", v, sync_dist=True)
        
    #     return
    
    def configure_optimizers(self):
        lr = 1e-4
        base_params = list(self.mask_former.parameters()) + \
                      list(self.mask_decoder.parameters())
        pooler_params = list(self.text_embedder.pooler.parameters())
        scale_params = [self.loss_fn.contrastive.logit_scale]
        
        opt_lr = [
            {"params": base_params, "lr": lr},
            {"params": pooler_params, "lr": lr}, 
            {"params": scale_params, "lr": lr},
        ]
        
        optimizer = torch.optim.AdamW(
            opt_lr, 
            betas=(0.9, 0.995),
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }]

            
    @torch.no_grad()
    def log_images(self, batch, split="train", pl_module=None):
        images, masks, ds_ids, c_ids = self.get_input(batch)
        img_embed, feats = self.encode_image(images) # B*N, d, H, W
 
        text_embeddings = self.text_embedder(ds_ids, c_ids) # B*N*M
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]
        text_embeddings = text_embeddings.unsqueeze(1)
        feats = feats[-2::-1]
        output, text, attn_weights = self.mask_former(img_embed, text_embeddings, feats)
        logits = self.mask_decoder(output, text_embeddings)
        
        pred_masks = torch.sigmoid(logits)
        pred_masks = (pred_masks > 0.5).float()
        
        return {
            "inputs": images,
            "gts": masks,
            "outputs": pred_masks,
        }
