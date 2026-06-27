import torch
from torch import nn
from typing import Type
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from src.models.transformer import Attention
from src.models.modules import MLPBlock
from einops import rearrange
import numpy as np
import math

class ShuffleFormer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 0.5,
        use_ape: bool = False,
        attention_downsample_rate: int = 2,
        use_fusion: bool = False,
        window_size = [2, 4, 4],
        uns_size = [8, 16, 1],
        n_mlp_blocks: int = 2,
        in_chans = [448, 112, 56]
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert (img_size % patch_size) == 0
        self.depth = depth
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.use_ape = use_ape
        if self.use_ape:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim)
            )
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = None
        self.text_ema = nn.Parameter(torch.tensor(0.))
        
        self.layers = nn.ModuleList()
        for i, dim in enumerate(in_chans):
            self.layers.append(
                ShuffleAttention(
                    in_channel=dim,
                    embed_dim=embed_dim,
                    n_attn=depth,
                    n_mlp=n_mlp_blocks,
                    shuffle_size=window_size[i],
                    unshuffle_s=uns_size[i],
                    num_heads=num_heads,
                    use_fusion=use_fusion,
                    mlp_ratio=mlp_ratio,
                )
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif 'RMSNorm' in m.__class__.__name__: 
             if hasattr(m, 'weight') and m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)
             if hasattr(m, 'bias') and m.bias is not None:
                 nn.init.constant_(m.bias, 0)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
    
    def forward(self, x, text_embed, feats):
        x = x
        x = rearrange(x, "b c h w -> b (h w) c")
        attn_weights = []
        out_x = []
        text = text_embed.clone()
        if self.use_ape:
            x = self.pos_embed + x
        alpha = torch.exp(self.text_ema).clamp(0.0, 0.999)
        for i, layer in enumerate(self.layers):
            x, text, attn_weight = layer(x, text, feats[i], alpha)
            out_x.append(x)
            attn_weights.append(attn_weight)
            
        return x, text, attn_weights, out_x

    
class ShuffleAttention(nn.Module):
    def __init__(
        self,
        in_channel: int=256,
        embed_dim: int = 256,
        n_attn: int = 2,
        mlp_ratio: int = 4,
        shuffle_size: int = 2,
        unshuffle_s: int = 4,
        n_mlp: int = 2,
        num_heads: int =8,
        use_fusion: bool = True,
    ):
        super().__init__()
        ds_dim = embed_dim // (unshuffle_s ** 2)
        self.shuffle = TokenShuffle(dim=embed_dim, s=shuffle_size, n_blocks=n_mlp, use_fusion=use_fusion)
        self.mask_attn = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim= int(mlp_ratio * embed_dim),
                activation=nn.GELU
            ) for _ in range(n_attn)
        ])
        self.unshuffle = ChannelUnshuffle(embed_dim, s=unshuffle_s, n_blocks=n_mlp, use_fusion=use_fusion)
        self.attn_gate = AttenGate(x_dim=in_channel, g_dim=ds_dim)
        self.conv = nn.Sequential(
            AtrousSeparableConvolution(
                in_channels=ds_dim + in_channel,
                out_channels=embed_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
    def forward(self, x, text=None, feat=None, alpha=1.0):
        x = self.shuffle(x)
        if text is not None:
            x = torch.cat((text, x), dim=1)
        
        for layer in self.mask_attn:
            x = layer(x)
            x, new_text = x[:, 1:], x[:, 0]
            text = alpha * text + (1 - alpha) * new_text
        x = self.unshuffle(x)
        x, attn_weight = self.attn_gate(x=feat, g=x)
        x = self.conv(x)
        
        return x, text

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)

        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
        
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.self_attn(q=x, k=x, v=x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.act = act_layer()
    
    def forward(self, x) -> torch.Tensor: 
        return self.linear2(self.act(self.linear1(x)))
        
class TokenShuffle(nn.Module):
    def __init__(
        self,
        dim: int,
        s: int = 2, # shuffle_size
        n_blocks: int = 2,
        use_fusion: bool = False,
    ):
        super().__init__()
        self.s = s
        hidden_dim=int(4 * dim)
        self.mlp = MLP(in_dim=dim, hidden_dim=hidden_dim, out_dim=  dim // (s ** 2)) #
        self.fusion = nn.Identity()
        if use_fusion:
            self.fusion = nn.ModuleList([
                MLP(in_dim=dim, hidden_dim=hidden_dim, out_dim=dim) 
                for _ in range(n_blocks)
            ])
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) > 3:
            x = x.flatten(2).permute(0, 2, 1)
        B, N, C = x.shape 
        s = self.s
        x = self.mlp(x) # B, N, C // s^2 
        H = W = int(math.sqrt(N)) // s
        x = rearrange(x, "b (h s1 w s2) c -> b (h w) (c s1 s2)", h=H, w=W, s1=s, s2=s) # B, N // s^2, C
        for layer in self.fusion:
            x = layer(x)
        return x

class ChannelUnshuffle(nn.Module): # 
    def __init__(self, dim, s = 2, n_blocks=2, use_fusion=False):
        super().__init__()
        self.s = s
        scale = s ** 2
        hidden_dim = int(4 * dim)
        self.proj = MLP(in_dim=dim, hidden_dim=hidden_dim, out_dim=dim) 
        self.fusion = nn.Identity()
        if use_fusion:
            self.fusion = nn.ModuleList([
                MLP(in_dim=dim // scale, hidden_dim=hidden_dim, out_dim=dim // scale) 
                for _ in range(n_blocks)
            ])
        
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.proj(x) # B, N, C
        x = rearrange(x, "b (h w) (c s1 s2) -> b (h s1 w s2) c", h = H, w = W, s1 = self.s, s2 = self.s) # B, N * s ^2, C // s^2
        for layer in self.fusion:
            x = layer(x)
        x = rearrange(x, "b (h w) c -> b c h w", h = H * self.s, w = W * self.s)
    
        return x
    
class AttenGate(nn.Module):
    def __init__(self, x_dim, g_dim, int_dim = None,):
        super().__init__()
        if int_dim is None:
            int_dim = (x_dim + g_dim) // 2
        self.proj_x = nn.Conv2d(x_dim, int_dim, 1, bias=False) # conv for feature map
        self.proj_g = nn.Conv2d(g_dim, int_dim, 1)
        self.gelu = nn.GELU()
        self.psi = nn.Conv2d(int_dim, 1, 1, bias=False)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor):
        x_org = x
        x = self.proj_x(x) # B, int_dim, Hx, Wx
        g_org = g
        g = self.proj_g(g)
        
        attn = self.psi(self.gelu(x + g)) # ψ(relu(x + g)) -> B, 1, Hg, Wg
        attn_weight = F.sigmoid(attn) #
        attn_out = (attn_weight * x_org)
        
        return torch.concat((attn_out, g_org), dim =1), attn_weight

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.body(x)
    

class Text2Mask(nn.Module):
    def __init__(self, init_temp = 10.):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_temp))
        self.bias = nn.Parameter(torch.zeros([]))
        
    def forward(self, img_feat: torch.Tensor, text_embed: torch.Tensor):
        H = W = int(math.sqrt(img_feat.shape[-1]))
        if len(img_feat.shape) > 3:
            B, C, H, W = img_feat.shape
            img_feat = rearrange(img_feat, "b c h w -> b (h w) c")
        img_feat = F.normalize(img_feat, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1) # B 1 C
        sim_map = torch.einsum("bnc, bmc -> bnm", text_embed, img_feat)
        sim_map = rearrange(sim_map, "b n (h w) -> b n h w", h=H, w=W)
        logit_scale = self.logit_scale.exp().clamp(max= 100)
        logits = sim_map * logit_scale + self.bias
        return logits
    