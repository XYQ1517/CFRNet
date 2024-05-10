import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class MB_Block(nn.Module):
    def __init__(self, dim, expansion=4):
        super(MB_Block, self).__init__()
        hidden_dim = int(dim * expansion)

        layers = OrderedDict()

        # expand
        expand_conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
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
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

        # c_dim_in = dim // 4
        #
        # self.dilate1 = nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, dilation=1, padding=1)
        # self.dilate2 = nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, dilation=2, padding=2)
        # self.dilate3 = nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, dilation=4, padding=4)
        # self.dilate4 = nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, dilation=8, padding=8)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        x = x + self.block(x)
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # x1 = self.dilate1(x1)
        # x2 = self.dilate2(x2)
        # x3 = self.dilate3(x3)
        # x4 = self.dilate4(x4)
        # x = torch.cat([x1, x2, x3, x4], dim=1) + x
        return x


class PatchExpand(nn.Module):
    def __init__(self, in_chan, out_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = in_chan
        self.expand = nn.Linear(in_chan, 4 * out_dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(2 * out_dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        _, H, W, _ = x.shape
        x = self.norm(x)
        x = x.view(B, -1, C//4)

        x = x.view(B, H, W, C//4)
        x = rearrange(x, 'b h w c-> b c h w')

        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


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


class TAFF(nn.Module):
    def __init__(self, inter_dim, embed_dims=[512, 256, 128], rfb=False):
        super(TAFF, self).__init__()
        self.inter_dim = inter_dim

        self.conv1 = self.add_conv(embed_dims[2], self.inter_dim, 1, 1)
        self.conv2 = self.add_conv(embed_dims[1], self.inter_dim, 1, 1)
        self.conv3 = self.add_conv(embed_dims[0], self.inter_dim, 1, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_1 = self.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_2 = self.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_3 = self.add_conv(self.inter_dim, compress_c, 1, 1)

        self.weights = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)

    def add_conv(self, in_ch, out_ch, ksize, stride, leaky=True):
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                           out_channels=out_ch, kernel_size=ksize, stride=stride,
                                           padding=pad, bias=False))
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        if leaky:
            stage.add_module('leaky', nn.LeakyReLU(0.1))
        else:
            stage.add_module('relu6', nn.ReLU6(inplace=True))
        return stage

    def forward(self, x1, x2, x3):
        x1_resized = self.conv1(x1)

        x2 = self.conv2(x2)
        x2_resized = F.interpolate(x2, scale_factor=2, mode='bilinear')

        x3 = self.conv3(x3)
        x3_resized = F.interpolate(x3, scale_factor=4, mode='bilinear')

        x1_weight = self.weight_1(x1_resized)
        x2_weight = self.weight_2(x2_resized)
        x3_weight = self.weight_3(x3_resized)
        out_weights = torch.cat((x1_weight, x2_weight, x3_weight), 1)
        out_weights = self.weights(out_weights)
        out_weights = F.softmax(out_weights, dim=1)

        out = x1_resized * out_weights[:, 0:1, :, :] + \
              x2_resized * out_weights[:, 1:2, :, :] + \
              x3_resized * out_weights[:, 2:, :, :]

        return out


class CFRNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, image_size=(256, 256),
                 depths=[3, 4, 6, 3], dims=[64, 128, 256, 512], stage_number=3):  # 96, 192, 384, 768
        super(CFRNet, self).__init__()
        self.stage_number = stage_number

        self.down_layer1 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.down_layer2 = nn.Sequential(
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
            )

        self.down_layer3 = nn.ModuleList()
        for i in range(self.stage_number):
            down_layer3 = nn.Sequential(
                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            )
            self.down_layer3.append(down_layer3)

        self.down_layer4 = nn.ModuleList()
        for i in range(self.stage_number):
            down_layer4 = nn.Sequential(
                LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
            )
            self.down_layer4.append(down_layer4)

        self.block_layer1 = nn.Sequential(
            *[MB_Block(dim=dims[0]) for j in range(depths[0])]
        )

        self.block_layer2 = nn.ModuleList()
        for i in range(self.stage_number):
            block_layer2 = nn.Sequential(
                *[MB_Block(dim=dims[1]) for j in range(depths[1])]
            )
            self.block_layer2.append(block_layer2)

        self.block_layer3 = nn.ModuleList()
        for i in range(self.stage_number):
            block_layer3 = nn.Sequential(
                *[MB_Block(dim=dims[2]) for j in range(depths[2])]
            )
            self.block_layer3.append(block_layer3)

        self.block_layer4 = nn.ModuleList()
        for i in range(self.stage_number):
            block_layer4 = nn.Sequential(
                *[MB_Block(dim=dims[3]) for j in range(depths[3])]
            )
            self.block_layer4.append(block_layer4)

        self.TAFF_layer = nn.ModuleList()
        for i in range(self.stage_number):
            TAFF_layer = TAFF(inter_dim=dims[1], embed_dims=[dims[3], dims[2], dims[1]])
            self.TAFF_layer.append(TAFF_layer)

        self.block_layer5 = nn.Sequential(
                *[MB_Block(dim=dims[0]) for j in range(depths[0])]
            )

        self.block_layer6 = nn.Sequential(
                *[MB_Block(dim=dims[1]) for j in range(depths[1])]
            )

        self.expand1 = PatchExpand(dims[1], dims[0])

        self.pred = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[1], 32, kernel_size=1),  # have head
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down_layer1(x)
        x0 = self.block_layer1(x)
        x1 = self.down_layer2(x0)

        inner_out = x1
        for i in range(self.stage_number):
            c1 = self.block_layer2[i](inner_out)

            x2 = self.down_layer3[i](c1)
            c2 = self.block_layer3[i](x2)

            x3 = self.down_layer4[i](c2)
            c3 = self.block_layer4[i](x3)
            inner_out = self.TAFF_layer[i](c1, c2, c3) + inner_out
        out = self.expand1(inner_out)
        out = self.block_layer5(out)
        out = self.block_layer6(torch.cat((out, x0), dim=1))
        out = self.pred(out)
        return out


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = CFRNet(image_size=(256, 256))
    y = model(x)
    print(y.shape)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True,
                                              print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
