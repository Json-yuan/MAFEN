#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Project ：新建文件夹 
@Modify Time 2023/8/5 10:30 
@Author     Shuiyuan Yang
@Version    1.0
@Desciption Tnt AND  Swin 双分支的网络结构 .953015873015873
0.1*loss0 +0.4*loss1 +0.4*loss2 + 0.1*loss3
"""
import math

import torch
import torch.nn as nn
import torch.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights,PatchMerging
from functools import partial
from tnt import tnt_b_patch16_224


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class Patch(nn.Module):

    def __init__(self, in_channels, hidden_dim, patch_size, drop_ratio=0):
        super(Patch, self).__init__()
        # self.pos_embedding = nn.Parameter(torch.empty(1, 196, hidden_dim).normal_(std=0.02))
        self.proj = nn.Conv2d(in_channels,hidden_dim, patch_size,patch_size)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        

    def forward(self,x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.pos_drop(x + self.pos_embedding)
        x = self.pos_drop(x)
        return x
class Conv2bnrl(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(Conv2bnrl, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class TokenEmb(nn.Module):
    """
    用于映射向量
    """
    def __init__(self,before_dim,after_dim,act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6,)):
        super(TokenEmb, self).__init__()
        self.before_dim = before_dim
        self.after_dim = after_dim
        self.norm_layer= norm_layer(before_dim,eps=1e-5, elementwise_affine=True)
        self.act_layer = act_layer()
        self.emd = nn.Linear(before_dim, after_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.act_layer(x)
        x = self.emd(x)
        return x


class swintnt(nn.Module):
    def __init__(self, token_dim = 640, num_classes=45):
        super(swintnt, self).__init__()
        self.hidden_dim = token_dim
        self.vit = tnt_b_patch16_224()
        weights_dict = torch.load(r"./weights/tnt_b_82.9.pth.tar", map_location='cpu')
        self.vit.load_state_dict(weights_dict, strict=False)
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.tc1 = TokenEmb(token_dim, 384)
        self.ct1 = TokenEmb(384, token_dim)
        self.tc2 = TokenEmb(token_dim, 384)
        self.ct2 = TokenEmb(384, token_dim)
        self.tc3 = TokenEmb(token_dim, 384)
        self.ct3 = TokenEmb(384, token_dim)

        self.downsample1_1 = PatchMerging(96)
        self.downsample1_2 = PatchMerging(192)
        self.downsample2_1 = PatchMerging(192)

        


        self.cross1 = CrossAttentionBlock(384, 16)
        self.cross2 = CrossAttentionBlock(384, 16)  
        self.cross3 = CrossAttentionBlock(384, 16)

        self.cross1_2 = CrossAttentionBlock(640, 16)
        self.cross1_3 = CrossAttentionBlock(640, 16)

        # self.patch1 = Patch(96, token_dim, 4)
        # self.patch2 = Patch(192, token_dim, 2)
        # self.patch3 = Patch(384, token_dim, 1)


        self.swinLinear = nn.Linear(768, token_dim)
        self.bn = nn.BatchNorm1d(token_dim *2 )
        self.bn = nn.BatchNorm1d(token_dim *2 )
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(token_dim *2, num_classes)
        self.head2= nn.Linear(token_dim,num_classes)
        self.head3= nn.Linear(768, num_classes)
        self.head4= nn.Linear(token_dim,num_classes)
        # init_weights()

    def _vit_process_input(self, x: torch.Tensor):
        '''
            vit encoder前的操作
        '''
        B = x.shape[0]
        inner_tokens = self.vit.patch_embed(x) + self.vit.inner_pos  # B*N, 8*8, C

        outer_tokens = self.vit.proj_norm2(self.vit.proj(self.vit.proj_norm1(inner_tokens.reshape(B, self.vit.num_patches, -1))))
        outer_tokens = torch.cat((self.vit.cls_token.expand(B, -1, -1), outer_tokens), dim=1)

        outer_tokens = outer_tokens + self.vit.outer_pos
        outer_tokens = self.vit.pos_drop(outer_tokens)
        return inner_tokens,outer_tokens
    def _mutil_level_fusion(self, cla1,cla2,cla3):
        "聚合多层token"
        x = self.cross1_3(torch.cat((self.cross1_2(torch.cat((cla1,cla2),dim=1)),cla3), dim=1))
        return x
    def forward(self,x):
        B = x.shape[0]
        inner_tokens,outer_tokens = self._vit_process_input(x)

        inner_tokens,outer_tokens = self.vit.blocks[0](inner_tokens,outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[1](inner_tokens, outer_tokens)
        s = self.swin.features[0:2](x)
        # stage1

        # cla1 = outer_tokens[:, 0:1, :] = self.cross1(torch.cat((outer_tokens[:, 0:1, ...], self.patch1(s.permute(0, 3, 1, 2))), dim=1))
        cla1_1 = outer_tokens[:, 0:1, :].clone()
        cla1_2 = self.tc1(cla1_1) 
        swin_1 = self.downsample1_1(s)
        swin_1 = self.downsample1_2(swin_1).reshape(B,196,384)
        cla1_3 = self.cross1(torch.cat((cla1_2,swin_1),dim=1))
        cla1 = outer_tokens[:, 0:1, :] = self.ct1(cla1_3)

        inner_tokens, outer_tokens = self.vit.blocks[2](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[3](inner_tokens, outer_tokens)
        s = self.swin.features[2:4](s)
        # stage2

        # cla2 = outer_tokens[:, 0:1, :] = self.cross2(torch.cat((outer_tokens[:, 0:1, ...], self.patch2(s.permute(0, 3, 1, 2))), dim=1))
        cla2_1 = outer_tokens[:, 0:1, :].clone()
        cla2_2 = self.tc2(cla2_1)
        swin_2 = self.downsample2_1(s).reshape(B,196,384)
        cla2_3 = self.cross2(torch.cat((cla2_2, swin_2),dim=1))
        cla2 = outer_tokens[:, 0:1, :] = self.ct2(cla2_3)

        inner_tokens, outer_tokens = self.vit.blocks[4](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[5](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[6](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[7](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[8](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[9](inner_tokens, outer_tokens)
        s = self.swin.features[4:6](s)
        # stage3

        # cla3 = outer_tokens[:, 0:1, :] = self.cross3(torch.cat((outer_tokens[:, 0:1, ...], self.patch3(s.permute(0, 3, 1, 2))), dim=1))
        cla3_1 = outer_tokens[:, 0:1, :].clone()
        cla3_2 = self.tc3(cla3_1)
        cla3_3 = self.cross3(torch.cat((cla3_2,s.reshape(B,196,384)),dim=1))
        
        cla3 = outer_tokens[:, 0:1, :] = self.ct3(cla3_3)

        inner_tokens, outer_tokens = self.vit.blocks[10](inner_tokens, outer_tokens)
        inner_tokens, outer_tokens = self.vit.blocks[11](inner_tokens, outer_tokens)
        s = self.swin.features[6:](s)



        s = self.swin.norm(s)
        s = self.swin.permute(s)
        s = self.swin.avgpool(s)
        s1 = self.swin.flatten(s)
        s = self.relu(self.swinLinear(s1))

        # tokens = torch.cat((cla1[:, 0], cla2[:, 0],cla3[:,0]),dim=1)
        
        tokens = self._mutil_level_fusion(cla1,cla2,cla3)
        x = torch.concat((s,outer_tokens[:,0]),dim=1)
        x = self.head(self.bn(x))
        tnt = self.head2(outer_tokens[:,0])
        swin = self.head3(s1)
        token = self.head4(tokens[:,0])
        return x, swin, tnt,token
        return x,s1,outer_tokens[:,0]

    def init_weights(self):
        self.patch1.apply(init_weights)
        self.patch2.apply(init_weights)
        self.patch3.apply(init_weights)
        self.cross1.apply(init_weights)
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)



