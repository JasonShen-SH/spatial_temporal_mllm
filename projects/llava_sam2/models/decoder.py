import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import pdb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class InternVLSimpleDecoder(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=8,
        img_size=448,
        in_chans=3,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_chans = in_chans
        self.decoder_embed_dim = decoder_embed_dim
        
        self.tokens_per_side = 16
        self.num_tokens = 256
        
        self.decoder_embed = nn.Linear(input_dim, decoder_embed_dim, bias=True)
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, decoder_embed_dim), 
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, 
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.patch_size_in_output = img_size // self.tokens_per_side
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            self.patch_size_in_output**2 * in_chans,
            bias=True
        )
        
        self.norm_pix_loss = False
        
        self.initialize_weights()
    
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.tokens_per_side,
            cls_token=False
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        将图像转换为patches
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size_in_output
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        B, N, _ = x.shape
        p = self.patch_size_in_output
        h = w = self.tokens_per_side
        
        x = x.reshape(shape=(B, h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(B, 3, h * p, w * p))
        
        return imgs

    def forward_loss(self, imgs, pred, mask=None):
        """
        计算重建损失
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0是keep，1是remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # pdb.set_trace()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss
    
    def forward(self, image_tokens, mask_tokens, target_imgs=None, mask=None):
        B = image_tokens.shape[0]
        image_features = self.decoder_embed(image_tokens)
        mask_features = self.decoder_embed(mask_tokens)
        
        x = torch.cat([image_features, mask_features], dim=1)
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        
        if self.training and target_imgs is not None:
            loss = self.forward_loss(target_imgs, pred, mask)
            imgs = self.unpatchify(pred)
            return imgs, loss
        else:
            imgs = self.unpatchify(pred)
            return imgs
    

def create_simple_internvl_decoder(lightweight=True):
    if lightweight:
        return InternVLSimpleDecoder(
            input_dim=4096,
            decoder_embed_dim=256,
            decoder_depth=4,
            decoder_num_heads=8,
            img_size=448,
        )
    else:
        return InternVLSimpleDecoder(
            input_dim=4096,
            decoder_embed_dim=512,
            decoder_depth=6,
            decoder_num_heads=8,
            img_size=448,
        )

def create_mae_style_mask_token(decoder_embed_dim):
    # mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
    mask_token = torch.randn(1, 1, 4096)
    torch.nn.init.normal_(mask_token, std=.02)
    return mask_token

# if __name__ == "__main__":
#     decoder_light = create_simple_internvl_decoder(lightweight=True)
#     decoder_standard = create_simple_internvl_decoder(lightweight=False)
    
#     print(f"轻量级模型参数量: {sum(p.numel() for p in decoder_light.parameters() if p.requires_grad):,}")
#     print(f"标准模型参数量: {sum(p.numel() for p in decoder_standard.parameters() if p.requires_grad):,}")
    
#     # mask_token_light = create_mae_style_mask_token(256)
#     mask_token_standard = create_mae_style_mask_token(512)
    
#     B = 2
#     image_tokens = torch.randn(B, 64, 4096)
    
#     # mask_tokens_light = mask_token_light.repeat(B, 192, 1)
#     mask_tokens_standard = mask_token_standard.repeat(B, 192, 1)
    
#     target_imgs = torch.randn(B, 3, 448, 448)
    
#     with torch.no_grad():
#         # output_light = decoder_light(image_tokens, mask_tokens_light)
#         # print(f"轻量级模型输出: {output_light.shape}")
        
#         output_standard = decoder_standard(image_tokens, mask_tokens_standard, target_imgs)
#         print(f"标准模型输出: {output_standard.shape}")
    
#     print("✅ 修正版decoder测试通过！")