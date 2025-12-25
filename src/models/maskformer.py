import torch
from torch import nn, Tensor
from typing import Type
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from src.models.transformer import Attention
from src.models.modules import PatchEmbed, MLPBlock, Conv2d_BN, Residual
from typing import Tuple, List
import math
 
class Upsample(nn.Module):
    def __init__(
        self,
        in_chans: int = 256,
        target_size: Tuple[int] = (64, 64),
    ):
        super().__init__()
        self.target_size = target_size
        
        self.conv1 = Residual(nn.Sequential(
            Conv2d_BN(in_chans, in_chans, 1, 1),
            nn.GELU(),
        ))
        
        self.conv2 = Residual(nn.Sequential(
            Conv2d_BN(in_chans, in_chans, 1, 1),
            nn.GELU(),
        ))
        
        self.conv3 = Conv2d_BN(in_chans, in_chans, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        size = (self.target_size[0] // 2, self.target_size[1] // 2)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        queries, keys = x.reshape(2, -1, *x.shape[-3:]).unbind(0)
        return queries, keys

    @torch.no_grad()
    def fuse(self):
        self.conv3 = self.final_conv.fuse()
        return self

class MaskFormer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_chans: int = 256,
        embed_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        activation: Type[nn.Module] = nn.GELU,
        use_ape: bool = True,
        rope_mixed: bool = False,
        attention_downsample_rate: int = 2,
        rope_theta:float = 100.,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert (img_size % patch_size) == 0
        self.depth = depth
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.2)
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=self.in_chans,
            embed_dim=embed_dim,
        ) 
        self.use_ape = use_ape
        if self.use_ape:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, (img_size // patch_size) ** 2 + 1, embed_dim
                )
            )
        else:
            self.pos_embed = None
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                MaskAttentionBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=self.mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        self.final_attn_image_to_token = Attention(
            self.embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(self.embed_dim)
        self.conv = nn.Sequential(
            Conv2d_BN(embed_dim, in_chans, 1),
            nn.GELU()
        )
        self.upsample = Upsample(in_chans=in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
    
    def forward(self, x, text_embed):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1,2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.use_ape:
            x = self.pos_embed + x
            
        queries = text_embed
        keys = x
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries = queries,
                keys = keys,
            )
        
        q = queries + text_embed # T2I + TEXT
        k = keys + x #I2T + Image
        
        attn_out = self.final_attn_image_to_token(q=k, k=q, v=queries)
        queries = queries + attn_out 
        x = torch.stack([queries, k], dim=0) # 2,B,N,C
        x = self.norm_final_attn(x)
        
        # upsample masking
        S, B, N, C = x.shape
        x = x.reshape(-1, N, C)
        x = x.permute(0, 2, 1).unsqueeze(-1) # B(N+1)C->BC(N+1)
        x = self.conv(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        cls_tokens = x[:, 0, :]
        q_cls, k_cls = cls_tokens.view(S, B, -1).unsqueeze(2).unbind(0)
        
        x = x[:, 1:, :] # 2B,N, C
        h = int(math.sqrt(x.shape[1]))
        x = x.reshape(-1, h, h, x.shape[-1]).permute(0, 3, 1, 2) # BCHW
        
        queries, keys =  self.upsample(x)
        return queries, keys, q_cls, k_cls
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m,  Conv2d_BN):
            m = self.m.fuse()
            return m

class MaskAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of dense
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )


    def forward(
        self, queries: Tensor, keys: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block 
        attn_out = self.self_attn(q=keys, k=keys, v=keys)
        keys = keys + attn_out
        keys = self.norm1(keys)

        # Cross attention block, tokens attending to image embedding 
        q = queries 
        k = keys 
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries
        k = keys
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
