import torch
from einops import rearrange
from torch import Tensor, nn

from src.models._base_model import BaseModel
from src.models.modules.misc import get_time_embedder
from src.utilities.utils import exists


RELU_LEAK = 0.2


class UNetBlock(torch.nn.Module):
    def __init__(
        self, in_chans, dim_out, time_emb_dim=None, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.0
    ):
        super().__init__()
        batch_norm = bn
        relu_leak = None if relu else RELU_LEAK
        kern_size = size
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        )

        ops = []
        # Next, the actual conv op
        if not transposed:
            # Regular conv
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=kern_size,
                    stride=2,
                    padding=pad,
                    bias=True,
                )
            )
        else:
            # Upsample and transpose conv
            ops.append(torch.nn.Upsample(scale_factor=2, mode="bilinear"))
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=(kern_size - 1),
                    stride=1,
                    padding=pad,
                    bias=True,
                )
            )
        # Finally, optional batch norm
        if batch_norm:
            ops.append(torch.nn.BatchNorm2d(dim_out))
        else:
            ops.append(nn.GroupNorm(8, dim_out))

        # Bundle ops into Sequential
        self.ops = torch.nn.Sequential(*ops)

        # First the activation
        if relu_leak is None or relu_leak == 0:
            self.act = torch.nn.ReLU()  # inplace=True)
        else:
            self.act = torch.nn.LeakyReLU(negative_slope=relu_leak)  # , inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb=None):
        x = self.ops(x)
        if exists(self.time_mlp):
            assert exists(time_emb), "Time embedding must be provided if time_mlp is not None"
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)

        return x


class UNet(BaseModel):
    def __init__(
        self,
        dim: int,
        with_time_emb: bool = False,
        outer_sample_mode: str = "bilinear",  # bilinear or nearest
        upsample_dims: tuple = (256, 256),  # (256, 256) or (128, 128)
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.outer_sample_mode = outer_sample_mode
        if upsample_dims is None:
            self.upsampler = nn.Identity()
        else:
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        in_channels = self.num_input_channels + self.num_conditional_channels
        # Build network operations
        if with_time_emb:
            # time embeddings
            self.time_dim = dim * 2
            self.time_emb_mlp = get_time_embedder(self.time_dim, dim, learned_sinusoidal_cond=False)
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        # ENCODER LAYERS
        self.init_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dropout_input = nn.Dropout(input_dropout)

        block_kwargs = dict(time_emb_dim=self.time_dim, dropout=dropout)
        self.input_ops = torch.nn.ModuleList(
            [
                UNetBlock(dim, dim * 2, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 2, dim * 2, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 2, dim * 4, transposed=False, bn=True, relu=False, **block_kwargs),
                UNetBlock(dim * 4, dim * 8, transposed=False, bn=True, relu=False, size=4, **block_kwargs),
                UNetBlock(dim * 8, dim * 8, transposed=False, bn=True, relu=False, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8, dim * 8, transposed=False, bn=False, relu=False, size=2, pad=0, **block_kwargs),
            ]
        )

        # DECODER LAYERS
        self.output_ops = torch.nn.ModuleList(
            [
                UNetBlock(dim * 8, dim * 8, transposed=True, bn=True, relu=True, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8 * 2, dim * 8, transposed=True, bn=True, relu=True, size=2, pad=0, **block_kwargs),
                UNetBlock(dim * 8 * 2, dim * 4, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 4 * 2, dim * 2, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 2 * 2, dim * 2, transposed=True, bn=True, relu=True, **block_kwargs),
                UNetBlock(dim * 2 * 2, dim, transposed=True, bn=True, relu=True, **block_kwargs),
            ]
        )
        self.readout = torch.nn.Sequential(
            # torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                in_channels=dim,  # * 2,
                out_channels=self.num_output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

        # Initialize weights
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _apply_ops(self, x: Tensor, time: Tensor = None):
        skip_connections = []
        # Encoder ops
        x = self.init_conv(x)
        x = self.dropout_input(x)
        for op in self.input_ops:
            x = op(x, time)
            skip_connections.append(x)
        # Decoder ops
        x = skip_connections.pop()
        for op in self.output_ops:
            x = op(x, time)
            if skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.readout(x)
        return x

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # Preprocess inputs for shape
        if self.num_conditional_channels > 0:
            x = torch.cat([inputs, condition], dim=1)
        else:
            x = inputs
            assert condition is None

        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) else None

        # Apply operations
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x)
        y = self._apply_ops(x, t)
        y = torch.nn.functional.interpolate(y, size=orig_x_shape, mode=self.outer_sample_mode)

        return y
