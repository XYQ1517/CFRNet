import math
import torch
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


def conv_3x3_bn(in_c, out_c, image_size, downsample=False):
    stride = 2 if downsample else 1
    layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU()
    )
    return layer


class MBConv_Encoder(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 downsample=False,
                 expansion=4):
        super(MBConv_Encoder, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(in_c * expansion)

        if self.downsample:
            # 只有第一层的时候，进行下采样
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        layers = OrderedDict()

        # expand
        expand_conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"expand_conv": expand_conv})

        # Depwise Conv
        dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"dw_conv": dw_conv})

        # project
        pro_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.block(x)
        else:
            return x + self.block(x)


class MBConv_Decoder(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 upsample=False,
                 expansion=4):
        super(MBConv_Decoder, self).__init__()
        self.upsample = upsample
        hidden_dim = int(out_c * expansion)

        if self.upsample:
            # 只有第一层的时候，进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.proj = nn.ConvTranspose2d(in_c, out_c, 1, 1, 0, bias=False)

        layers = OrderedDict()

        # expand
        expand_conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"expand_conv": expand_conv})

        # Depwise Conv
        dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"dw_conv": dw_conv})

        # project
        pro_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.upsample:
            return self.proj(self.up(x)) + self.block(self.up(x))
        else:
            return x + self.block(x)


class Attention(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == in_c)

        self.ih, self.iw = image_size if len(image_size) == 2 else (image_size, image_size)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads)
        )

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)

        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(in_c, inner_dim * 3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, out_c),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # [q,k,v]
        qkv = self.qkv(x).chunk(3, dim=-1)
        # q,k,v:[batch_size, num_heads, num_patches, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # [batch_size, num_heads, ih*iw, ih*iw]
        # 时间复杂度：O(图片边长的平方)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Transformer_Encoder(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 downsample=False,
                 dropout=0.,
                 expansion=4,
                 norm_layer=nn.LayerNorm):
        super(Transformer_Encoder, self).__init__()
        self.downsample = downsample
        hidden_dim = int(in_c * expansion)
        self.ih, self.iw = image_size

        if self.downsample:
            # 第一层进行下采样
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        self.attn = Attention(in_c, out_c, image_size, heads, dim_head, dropout)
        self.ffn = MLP(out_c, hidden_dim)
        self.norm1 = norm_layer(in_c)
        self.norm2 = norm_layer(out_c)

    def forward(self, x):
        x1 = self.pool1(x) if self.downsample else x
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.attn(self.norm1(x1))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        x2 = self.proj((self.pool2(x))) if self.downsample else x

        x3 = x1 + x2
        _, _, H2, W2 = x3.shape
        x4 = rearrange(x3, 'b c h w -> b (h w) c')
        x4 = self.ffn(self.norm2(x4), H2, W2)
        x4 = rearrange(x4, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        out = x3 + x4
        return out


class Transformer_Decoder(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 upsample=False,
                 dropout=0.,
                 expansion=4,
                 norm_layer=nn.LayerNorm):
        super(Transformer_Decoder, self).__init__()
        self.upsample = upsample
        hidden_dim = int(in_c * expansion)
        self.ih, self.iw = image_size

        if self.upsample:
            # 第一层进行下采样
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.proj = nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False)

        self.attn = Attention(in_c, out_c, image_size, heads, dim_head, dropout)
        self.ffn = MLP(out_c, hidden_dim)
        self.norm1 = norm_layer(in_c)
        self.norm2 = norm_layer(out_c)

    def forward(self, x):
        x1 = self.up1(x) if self.upsample else x
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.attn(self.norm1(x1))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        x2 = self.proj((self.up2(x))) if self.upsample else x

        x3 = x1 + x2
        _, _, H2, W2 = x3.shape
        x4 = rearrange(x3, 'b c h w -> b (h w) c')
        x4 = self.ffn(self.norm2(x4), H2, W2)
        x4 = rearrange(x4, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        out = x3 + x4
        return out


class HCTNet_Encoder(nn.Module):
    def __init__(self,
                 image_size=(256, 256),
                 in_channels: int = 3,
                 num_blocks: list = [2, 2, 3, 5, 2],  # L [2, 2, 6, 14, 2]
                 channels: list = [64, 64, 128, 256, 512],  # D
                 num_classes: int = 1,
                 block_types=['C', 'C', 'T', 'T'],
                 drop_rate=0.):
        super(HCTNet_Encoder, self).__init__()

        assert len(image_size) == 2, "image size must be: {H,W}"
        assert len(channels) == 5
        assert len(block_types) == 4

        ih, iw = image_size
        block = {'C': MBConv_Encoder, 'T': Transformer_Encoder}

        self.decoder_dim = channels[4]

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2)
        )
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)
        )
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8)
        )
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16)
        )
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32)
        )

        # 总共下采样32倍 2^5=32
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.s0(x)
        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        c4 = self.s4(c3)

        return c1, c2, c3, c4

    def _make_layer(self, block, in_c, out_c, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(in_c, out_c, image_size, downsample=True))
            else:
                layers.append(block(out_c, out_c, image_size, downsample=False))
        return nn.Sequential(*layers)


class HCTNet_Decoder(nn.Module):
    def __init__(self,
                 block_id,
                 in_channel,
                 out_channel,
                 image_size=(256, 256),
                 num_blocks: list = [2, 2, 3, 5, 2],  # [2, 2, 5, 12, 2]
                 block_types=['C', 'C', 'T', 'T'],
                 drop_rate=0.):
        super(HCTNet_Decoder, self).__init__()

        assert len(image_size) == 2, "image size must be: {H,W}"
        assert len(block_types) == 4

        ih, iw = image_size
        block = {'C': MBConv_Decoder, 'T': Transformer_Decoder}

        self.block_id = block_id

        if self.block_id == 1:
            self.layer = self._make_layer(
                block[block_types[0]], in_channel, out_channel, num_blocks[1], (ih // 2, iw // 2)
            )
        elif self.block_id == 2:
            self.layer = self._make_layer(
                block[block_types[1]], in_channel, out_channel, num_blocks[2], (ih // 4, iw // 4)
            )
        elif self.block_id == 3:
            self.layer = self._make_layer(
                block[block_types[2]], in_channel, out_channel, num_blocks[3], (ih // 8, iw // 8)
            )
        elif self.block_id == 4:
            self.layer = self._make_layer(
                block[block_types[3]], in_channel, out_channel, num_blocks[4], (ih // 16, iw // 16)
            )

        self.conv = nn.Sequential(
                nn.Conv2d(in_channel * 2, out_channel * 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        return x

    def _make_layer(self, block, in_c, out_c, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(in_c, out_c, image_size, upsample=True))
            else:
                layers.append(block(out_c, out_c, image_size, upsample=False))
        return nn.Sequential(*layers)


class LinearMLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class HCTNet(nn.Module):
    def __init__(self, image_size=(256, 256), num_classes=1, embed_dims=[256, 128, 64, 32], decoder_dim=256, drop_rate=0.):
        super(HCTNet, self).__init__()
        self.image_size = image_size
        self.backbone = HCTNet_Encoder(image_size)  # [64, 128, 320, 512]

        self.center = nn.Sequential(
              nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
              nn.BatchNorm2d(1024),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
              nn.ConvTranspose2d(1024, 512, 3, 1, 1, bias=False)
            )

        self.decoder1 = HCTNet_Decoder(4, 512, 256, self.image_size)
        self.decoder2 = HCTNet_Decoder(3, 256, 128, self.image_size)
        self.decoder3 = HCTNet_Decoder(2, 128, 64, self.image_size)
        self.decoder4 = HCTNet_Decoder(1, 64, 32, self.image_size)

        # segmentation head
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)
        self.linear_fuse_bn = nn.BatchNorm2d(decoder_dim)
        self.dropout = nn.Dropout2d(drop_rate)

        self.linear_pred = nn.Sequential(
            nn.Conv2d(decoder_dim, 32, kernel_size=1),  # have head
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4 = self.backbone(x)

        center = self.center(x4)

        # Decoder
        c1 = self.decoder1(torch.cat((center, x4), 1))
        c2 = self.decoder2(torch.cat((c1, x3), 1))
        c3 = self.decoder3(torch.cat((c2, x2), 1))
        c4 = self.decoder4(torch.cat((c3, x1), 1))
        # out = self.linear_pred(c4)  # no head

        # Head
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.linear_fuse_bn(_c)

        out = self.dropout(_c)
        out = self.linear_pred(out)
        return out


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = HCTNet(image_size=(256, 256))
    y = model(x)
    print(y.shape)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
