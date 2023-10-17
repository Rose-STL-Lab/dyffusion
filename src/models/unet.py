from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn

from src.models._base_model import BaseModel
from src.models.modules.attention import Attention, LinearAttention
from src.models.modules.misc import Residual, get_time_embedder
from src.models.modules.net_norm import PreNorm
from src.utilities.utils import default, exists


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout: float = 0.0):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        double_conv_layer: bool = True,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups, dropout=dropout1)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout2) if double_conv_layer else nn.Identity()
        self.residual_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.residual_conv(x)


# model
class Unet(BaseModel):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        num_conditions: int = 0,
        resnet_block_groups=8,
        with_time_emb: bool = False,
        block_dropout: float = 0.0,  # for second block in resnet block
        block_dropout1: float = 0.0,  # for first block in resnet block
        attn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        double_conv_layer: bool = True,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 128)
        keep_spatial_dims: bool = False,
        init_kernel_size: int = 7,
        init_padding: int = 3,
        init_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # determine dimensions
        input_channels = self.num_input_channels + self.num_conditional_channels
        output_channels = self.num_output_channels or input_channels
        self.save_hyperparameters()

        if num_conditions >= 1:
            assert (
                self.num_conditional_channels > 0
            ), f"num_conditions is {num_conditions} but num_conditional_channels is {self.num_conditional_channels}"
        #     input_channels += input_channels * num_conditions

        init_dim = default(init_dim, dim)
        assert (upsample_dims is None and outer_sample_mode is None) or (
            upsample_dims is not None and outer_sample_mode is not None
        ), "upsample_dims and outer_sample_mode must be both None or both not None"
        if outer_sample_mode is not None:
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        else:
            self.upsampler = None

        self.init_conv = nn.Conv2d(
            input_channels, init_dim, init_kernel_size, padding=init_padding, stride=init_stride
        )
        self.dropout_input = nn.Dropout(input_dropout)
        self.dropout_input_for_residual = nn.Dropout(input_dropout)

        if with_time_emb:
            self.time_dim = dim * 2  # 4
            self.time_emb_mlp = get_time_embedder(self.time_dim, dim, learned_sinusoidal_cond, learned_sinusoidal_dim)
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            dropout2=block_dropout,
            dropout1=block_dropout1,
            double_conv_layer=double_conv_layer,
            time_emb_dim=self.time_dim,
        )
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            do_downsample = not is_last and not keep_spatial_dims

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        Residual(
                            PreNorm(
                                dim_in, fn=LinearAttention(dim_in, rescale="qkv", dropout=attn_dropout), norm=LayerNorm
                            )
                        ),
                        Downsample(dim_in, dim_out) if do_downsample else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, fn=Attention(mid_dim, dropout=attn_dropout), norm=LayerNorm))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            do_upsample = not is_last and not keep_spatial_dims

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out),
                        block_klass(dim_out + dim_in, dim_out),
                        Residual(
                            PreNorm(
                                dim_out,
                                fn=LinearAttention(dim_out, rescale="qkv", dropout=attn_dropout),
                                norm=LayerNorm,
                            )
                        ),
                        Upsample(dim_out, dim_in) if do_upsample else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = input_channels * (1 if not learned_variance else 2)
        self.out_dim = default(output_channels, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = self.get_head()

        if hasattr(self, "spatial_shape") and self.spatial_shape is not None:
            b, s1, s2 = 1, *self.spatial_shape
            self.example_input_array = [
                torch.rand(b, self.num_input_channels, s1, s2),
                torch.rand(b) if with_time_emb else None,
                torch.rand(b, self.num_conditional_channels, s1, s2) if self.num_conditional_channels > 0 else None,
            ]

    def get_head(self):
        return nn.Conv2d(self.hparams.dim, self.out_dim, 1)

    def set_head_to_identity(self):
        self.final_conv = nn.Identity()

    def get_block(self, dim_in, dim_out, dropout: Optional[float] = None):
        return ResnetBlock(
            dim_in,
            dim_out,
            groups=self.hparams.resnet_block_groups,
            dropout1=dropout or self.hparams.block_dropout1,
            dropout2=dropout or self.hparams.block_dropout,
            time_emb_dim=self.time_dim,
        )

    def get_extra_last_block(self, dropout: Optional[float] = None):
        return self.get_block(self.hparams.dim, self.hparams.dim, dropout=dropout)

    def forward(self, x, time=None, condition=None, return_time_emb: bool = False):
        if self.num_conditional_channels > 0:
            # condition = default(condition, lambda: torch.zeros_like(x))
            x = torch.cat((condition, x), dim=1)
        else:
            assert condition is None, "condition is not None but num_conditional_channels is 0"

        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x) if exists(self.upsampler) else x
        x = self.init_conv(x)
        r = self.dropout_input_for_residual(x) if self.hparams.input_dropout > 0 else x.clone()
        x = self.dropout_input(x)

        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) else None

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        if exists(self.upsampler):
            # x = F.interpolate(x, orig_x_shape, mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)

        if return_time_emb:
            return x, t
        return x
