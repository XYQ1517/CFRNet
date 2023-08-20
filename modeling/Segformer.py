import torch.nn.functional as F
from typing import Tuple, Iterable
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
from timm.models.layers import trunc_normal_

Tuple4i = Tuple[int, int, int, int]


def drop_path(x, p: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if p == 0. or not training:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
        nn.init.normal_(m.weight, std=(2.0 / fan_out) ** 0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MixFeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Depth-wise convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            sr_ratio_tuple = (sr_ratio, sr_ratio)
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 dropout_p: float = 0.0, drop_path_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.drop_path_p = drop_path_p
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFeedForward(dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)

    def forward(self, x, h, w):
        skip = x
        x = self.norm1(x)
        x = self.attn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ffn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w


class MixTransformerStage(nn.Module):
    def __init__(
        self,
        patch_embed: OverlapPatchEmbed,
        blocks: Iterable[TransformerBlock],
        norm: nn.LayerNorm,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class MixTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple4i = (64, 128, 256, 512),
        num_heads: Tuple4i = (1, 2, 4, 8),
        qkv_bias: bool = False,
        dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        depths: Tuple4i = (3, 4, 6, 3),
        sr_ratios: Tuple4i = (8, 4, 2, 1),
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        for l in range(len(depths)):
            blocks = [
                TransformerBlock(dim=embed_dims[l], num_heads=num_heads[l], qkv_bias=qkv_bias,
                                 dropout_p=dropout_p, sr_ratio=sr_ratios[l],
                                 drop_path_p=drop_path_p * (sum(depths[:l])+i) / (sum(depths)-1))
                for i in range(depths[l])
            ]
            if l == 0:
                patch_embed = OverlapPatchEmbed((7, 7), stride=4, in_chans=in_chans,
                                                embed_dim=embed_dims[l])
            else:
                patch_embed = OverlapPatchEmbed((3, 3), stride=2, in_chans=embed_dims[l - 1],
                                                embed_dim=embed_dims[l])
            norm = nn.LayerNorm(embed_dims[l], eps=1e-6)
            self.stages.append(MixTransformerStage(patch_embed, blocks, norm))

        self.init_weights()

    def init_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def _mit_bx(embed_dims: Tuple4i, depths: Tuple4i) -> MixTransformer:
    return MixTransformer(
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        qkv_bias=True,
        depths=depths,
        sr_ratios=(8, 4, 2, 1),
        dropout_p=0.0,
        drop_path_p=0.1,
    )


def mit_b0():
    return _mit_bx(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2))


def mit_b1():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2))


def mit_b2():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3))


def mit_b3():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3))


def mit_b4():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 8, 27, 3))


def mit_b5():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3))


class LinearMLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Segformer(nn.Module):
    def __init__(self, num_classes=1, embed_dims=[64, 128, 320, 512], decoder_dim=768, drop_rate=0.):
        super(Segformer, self).__init__()
        self.backbone = mit_b2()  # [64, 128, 320, 512]

        # segmentation head
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)
        self.linear_fuse_bn = nn.BatchNorm2d(decoder_dim)
        self.dropout = nn.Dropout2d(drop_rate)
        self.linear_pred = nn.Conv2d(decoder_dim, 64, kernel_size=1)

        self.conv_last = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pvt = self.backbone(x)
        c1 = pvt[0]  # [-1, 64, H/4, W/4]
        c2 = pvt[1]  # [-1, 128, H/8, W/8]
        c3 = pvt[2]  # [-1, 320, H/16, W/16]
        c4 = pvt[3]  # [-1, 512, H/32, W/32]

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # h_out, w_out = c1.size()[2] * 4, c1.size()[3] * 4

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.linear_fuse_bn(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        # x = F.interpolate(input=x, size=(h_out, w_out), mode='bilinear', align_corners=False)
        x = self.conv_last(x)
        x = x.type(torch.float32)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Segformer()
    y = model(x)
    print(y.shape)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
