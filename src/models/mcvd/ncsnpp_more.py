# adding conditional group-norm as per https://arxiv.org/pdf/2105.05233.pdf
import functools
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: skip-file
from src.models.mcvd import layers, layerspp

from .._base_model import BaseModel


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppGN
get_act = layers.get_act
default_initializer = layers.default_init


class NCSNpp(BaseModel):
    """NCSN++ model"""

    def __init__(
        self,
        architecture: str = "unetmore",
        dim: int = 64,
        n_head_channels: int = 64,
        # number of channels per self-attention head (should ideally be larger or equal to dim, otherwise you may have a size mismatch error)
        dim_mults: Sequence[int] = (1, 2, 3, 4),
        resnet_block_groups: int = 2,
        attn_resolutions: Sequence[int] = (8, 16, 32),
        dropout: float = 0.0,
        with_time_emb: bool = True,
        cond_emb: bool = False,
        num_conditions: int = 1,
        output_all_frames: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.act = act = nn.SiLU()  # get_act(config)
        # self.register_buffer('sigmas', get_sigmas(config))
        self.is3d = architecture in ["unetmore3d", "unetmorepseudo3d"]
        self.pseudo3d = architecture == "unetmorepseudo3d"
        if self.is3d:
            from . import layers3d

        self.channels = self.num_input_channels
        self.num_frames = num_frames = self.num_output_channels  #  config.data.num_frames
        self.num_frames_cond = (
            self.num_conditional_channels
        )  # num_conditions  #config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0)
        self.n_frames = num_frames + self.num_frames_cond
        num_input_channels = self.num_input_channels + self.num_conditional_channels

        self.nf = nf = dim * self.n_frames if self.is3d else dim  # We must prevent problems by multiplying by n_frames
        self.numf = numf = (
            dim * self.num_frames if self.is3d else dim
        )  # We must prevent problems by multiplying by n_frames

        self.num_res_blocks = resnet_block_groups
        self.attn_resolutions = attn_resolutions

        resamp_with_conv = True
        self.num_resolutions = num_resolutions = len(dim_mults)
        self.all_resolutions = all_resolutions = [self.spatial_shape[0] // (2**i) for i in range(num_resolutions)]

        self.with_time_emb = with_time_emb  # noise-conditional
        self.cond_emb = cond_emb
        fir = True
        fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = skip_rescale = True
        self.resblock_type = resblock_type = "biggan"
        self.embedding_type = embedding_type = "positional"
        self.init_scale = init_scale = 0.0
        assert embedding_type in ["fourier", "positional"]

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=16))
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        temb_dim = None

        if self.with_time_emb:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            temb_dim = nf * 4

            if self.cond_emb:
                modules.append(
                    torch.nn.Embedding(num_embeddings=2, embedding_dim=nf // 2)
                )  # makes it 8 times smaller (16 if ngf=32) since it should be small because there are only two possible values:
                temb_dim += nf // 2

        if self.pseudo3d:
            conv3x3 = functools.partial(
                layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.n_frames, act=self.act
            )  # Activation here as per https://arxiv.org/abs/1809.04096
        elif self.is3d:
            conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.n_frames)
        else:
            conv3x3 = layerspp.conv3x3

        if self.is3d:
            AttnBlockDown = functools.partial(
                layers3d.AttnBlockpp3d,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                n_head_channels=n_head_channels,
                n_frames=self.n_frames,
                act=None,
            )  # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
            AttnBlockUp = functools.partial(
                layers3d.AttnBlockpp3d,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                n_head_channels=n_head_channels,
                n_frames=self.num_frames,
                act=None,
            )  # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
        else:
            AttnBlockDown = AttnBlockUp = functools.partial(
                layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale, n_head_channels=n_head_channels
            )

        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if resblock_type == "ddpm":
            ResnetBlockDown = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                n_frames=self.n_frames,
                pseudo3d=self.pseudo3d,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096
            ResnetBlockUp = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                n_frames=self.num_frames,
                pseudo3d=self.pseudo3d,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096

        elif resblock_type == "biggan":
            ResnetBlockDown = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                n_frames=self.n_frames,
                pseudo3d=self.pseudo3d,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096
            ResnetBlockUp = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                n_frames=self.num_frames,
                pseudo3d=self.pseudo3d,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block
        modules.append(conv3x3(num_input_channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(resnet_block_groups):
                out_ch = nf * dim_mults[i_level]
                modules.append(ResnetBlockDown(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlockDown(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlockDown(down=True, in_ch=in_ch))

                hs_c.append(in_ch)

        # Middle Block
        in_ch = hs_c[-1]
        modules.append(ResnetBlockDown(in_ch=in_ch))
        modules.append(AttnBlockDown(channels=in_ch))
        if self.is3d:
            # Converter
            modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
            in_ch = int(in_ch * self.num_frames / self.n_frames)
        modules.append(ResnetBlockUp(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(resnet_block_groups + 1):
                out_ch = numf * dim_mults[i_level]
                if self.is3d:  # 1x1 self.num_frames + self.num_frames_cond -> self.num_frames
                    modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
                    in_ch_old = int(hs_c.pop() * self.num_frames / self.n_frames)
                else:
                    in_ch_old = hs_c.pop()
                modules.append(ResnetBlockUp(in_ch=in_ch + in_ch_old, out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlockUp(channels=in_ch))

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlockUp(in_ch=in_ch, up=True))

        assert not hs_c
        self.final_dim = in_ch
        modules.append(
            layerspp.get_act_norm(
                act=act, act_emb=act, norm="group", ch=self.final_dim, is3d=self.is3d, n_frames=self.num_frames
            )
        )
        modules.append(self.get_head())

        self.all_modules = nn.ModuleList(modules)

    def get_head(self):
        if self.pseudo3d:
            from . import layers3d

            conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act)
        elif self.is3d:
            from . import layers3d

            conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
        else:
            conv3x3_last = layerspp.conv3x3
        return conv3x3_last(self.final_dim, self.num_output_channels, init_scale=self.init_scale)

    def set_head_to_identity(self):
        self.all_modules[-1] = nn.Identity()

    def forward(self, x: Tensor, time: Tensor = None, condition: Tensor = None, cond_mask=None, return_time_emb=False):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if condition is not None:
            x = torch.cat([x, condition], dim=1)  # B, (num_frames+num_frames_cond)*C, H, W

        if self.is3d:  # B, N*C, H, W -> B, C*N, H, W : subtle but important difference!
            B, NC, H, W = x.shape
            CN = NC
            x = x.reshape(B, self.n_frames, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)

        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            # assert that all time elems are integers
            # assert torch.allclose(time, time.long()), f'All time elements must be integers, got {time}'
            timesteps = time  # .long()
            # used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.with_time_emb:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))  # b x k
            m_idx += 1
            if self.cond_emb:
                if cond_mask is None:
                    cond_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
                temb = torch.cat([temb, modules[m_idx](cond_mask)], dim=1)  # b x (k/8 + k)
                m_idx += 1
        else:
            temb = None

        # Downsampling block
        x = x.contiguous()
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                hs.append(h)

        # Middle Block

        # ResBlock
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        # AttnBlock
        h = modules[m_idx](h)
        m_idx += 1

        # Converter
        if (
            self.is3d
        ):  # downscale time-dim, we decided to do it here, but could also have been done earlier or at the end
            # B, C*(num_frames+num_cond), H, W -> B, C, (num_frames+num_cond), H, W -----conv1x1-----> B, C, num_frames, H, W -> B, C*num_frames, H, W
            B, CN, H, W = h.shape
            h = h.reshape(-1, self.n_frames, H, W)
            h = modules[m_idx](h)
            m_idx += 1
            h = h.reshape(B, -1, H, W)

        # ResBlock
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if self.is3d:
                    # Get h and h_old
                    B, CN, H, W = h.shape
                    h = h.reshape(B, -1, self.num_frames, H, W)
                    prev = hs.pop().reshape(-1, self.n_frames, H, W)
                    # B, C*Nhs, H, W -> B, C, Nhs, H, W -----conv1x1-----> B, C, Nh, H, W -> B, C*Nh, H, W
                    prev = modules[m_idx](prev).reshape(B, -1, self.num_frames, H, W)
                    m_idx += 1
                    # Concatenate
                    h_comb = torch.cat([h, prev], dim=1)  # B, C, N, H, W
                    h_comb = h_comb.reshape(B, -1, H, W)
                else:
                    prev = hs.pop()
                    h_comb = torch.cat([h, prev], dim=1)
                h = modules[m_idx](h_comb, temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs
        # GroupNorm
        h = modules[m_idx](h)
        m_idx += 1

        # conv3x3_last
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        if (
            self.hparams.output_all_frames and condition is not None
        ):  # we only keep the non-cond images (but we could use them eventually)
            _, h = torch.split(
                h,
                [self.num_frames_cond * self.config.data.channels, self.num_frames * self.config.data.channels],
                dim=1,
            )

        if self.is3d:  # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
            B, CN, H, W = h.shape
            NC = CN
            h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)

        if return_time_emb:
            return h, temb
        return h


class SPADE_NCSNpp(nn.Module):
    """NCSN++ model with SPADE normalization"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        # self.register_buffer('sigmas', get_sigmas(config))
        self.is3d = config.model.arch in ["unetmore3d", "unetmorepseudo3d"]
        self.pseudo3d = config.model.arch == "unetmorepseudo3d"
        if self.is3d:
            from . import layers3d

        self.channels = channels = config.data.channels
        self.num_frames = num_frames = config.data.num_frames
        self.num_frames_cond = num_frames_cond = config.data.num_frames_cond + getattr(
            config.data, "num_frames_future", 0
        )
        self.n_frames = num_frames

        self.nf = nf = (
            config.model.ngf * self.num_frames if self.is3d else config.model.ngf
        )  # We must prevent problems by multiplying by num_frames
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = getattr(config.model, "dropout", 0.0)
        resamp_with_conv = True
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2**i) for i in range(num_resolutions)]

        self.conditional = conditional = getattr(config.model, "time_conditional", True)  # noise-conditional
        self.cond_emb = getattr(config.model, "cond_emb", False)
        fir = True
        fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = skip_rescale = True
        self.resblock_type = resblock_type = "biggan"
        self.embedding_type = embedding_type = "positional"
        init_scale = 0.0
        assert embedding_type in ["fourier", "positional"]

        self.spade_dim = spade_dim = getattr(config.model, "spade_dim", 128)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.

            modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=16))
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        temb_dim = None

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            temb_dim = nf * 4

            if self.cond_emb:
                modules.append(
                    torch.nn.Embedding(num_embeddings=2, embedding_dim=nf // 2)
                )  # makes it 8 times smaller (16 if ngf=32) since it should be small because there are only two possible values:
                temb_dim += nf // 2

        if self.pseudo3d:
            conv3x3 = functools.partial(
                layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act
            )  # Activation here as per https://arxiv.org/abs/1809.04096
            conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_pseudo3d, n_frames=self.channels, act=self.act)
        elif self.is3d:
            conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
            conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_3d, n_frames=self.channels)
        else:
            conv3x3 = layerspp.conv3x3
            conv1x1_cond = layerspp.conv1x1

        if self.is3d:
            AttnBlock = functools.partial(
                layers3d.AttnBlockpp3d,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                n_head_channels=config.model.n_head_channels,
                n_frames=self.num_frames,
                act=None,
            )  # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
        else:
            AttnBlock = functools.partial(
                layerspp.AttnBlockpp,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                n_head_channels=config.model.n_head_channels,
            )

        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        ResnetBlockDDPM = layerspp.ResnetBlockDDPMppSPADE
        ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppSPADE

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                pseudo3d=self.pseudo3d,
                n_frames=self.num_frames,
                num_frames_cond=num_frames_cond,
                cond_ch=num_frames_cond * channels,
                spade_dim=spade_dim,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=temb_dim,
                is3d=self.is3d,
                pseudo3d=self.pseudo3d,
                n_frames=self.num_frames,
                num_frames_cond=num_frames_cond,
                cond_ch=num_frames_cond * channels,
                spade_dim=spade_dim,
                act3d=True,
            )  # Activation here as per https://arxiv.org/abs/1809.04096

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        modules.append(conv3x3(channels * self.num_frames, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                hs_c.append(in_ch)

        # Middle Block
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                in_ch_old = hs_c.pop()
                modules.append(ResnetBlock(in_ch=in_ch + in_ch_old, out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        modules.append(
            layerspp.get_act_norm(
                act=act,
                act_emb=act,
                norm="spade",
                ch=in_ch,
                is3d=self.is3d,
                n_frames=self.num_frames,
                num_frames_cond=num_frames_cond,
                cond_ch=num_frames_cond * channels,
                spade_dim=spade_dim,
                cond_conv=conv3x3,
                cond_conv1=conv1x1_cond,
            )
        )
        modules.append(conv3x3(in_ch, channels * self.num_frames, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, cond=None, cond_mask=None):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # if cond is not None:
        #   x = torch.cat([x, cond], dim=1) # B, (num_frames+num_frames_cond)*C, H, W

        if self.is3d:  # B, N*C, H, W -> B, C*N, H, W : subtle but important difference!
            B, NC, H, W = x.shape
            CN = NC
            x = x.reshape(B, self.num_frames, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)
            if cond is not None:
                B, NC, H, W = cond.shape
                CN = NC
                cond = (
                    cond.reshape(B, self.num_frames_cond, self.channels, H, W)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(B, CN, H, W)
                )

        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            # used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))  # b x k
            m_idx += 1
            if self.cond_emb:
                if cond_mask is None:
                    cond_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
                temb = torch.cat([temb, modules[m_idx](cond_mask)], dim=1)  # b x (k/8 + k)
                m_idx += 1
        else:
            temb = None

        # Downsampling block

        x = x.contiguous()
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, cond=cond)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1], cond=cond)
                else:
                    h = modules[m_idx](hs[-1], temb, cond=cond)
                m_idx += 1
                hs.append(h)

        # Middle Block

        # ResBlock
        h = hs[-1]
        h = modules[m_idx](h, temb, cond=cond)
        m_idx += 1
        # AttnBlock
        h = modules[m_idx](h)
        m_idx += 1

        # ResBlock
        h = modules[m_idx](h, temb, cond=cond)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if self.is3d:
                    # Get h and h_old
                    B, CN, H, W = h.shape
                    h = h.reshape(B, -1, self.num_frames, H, W)
                    prev = hs.pop().reshape(B, -1, self.num_frames, H, W)
                    # Concatenate
                    h_comb = torch.cat([h, prev], dim=1)  # B, C, N, H, W
                    h_comb = h_comb.reshape(B, -1, H, W)
                else:
                    prev = hs.pop()
                    h_comb = torch.cat([h, prev], dim=1)
                h = modules[m_idx](h_comb, temb, cond=cond)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h, cond=cond)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, cond=cond)
                    m_idx += 1

        assert not hs
        # GroupNorm
        h = modules[m_idx](h, cond=cond)
        m_idx += 1

        # conv3x3_last
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        if self.is3d:  # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
            B, CN, H, W = h.shape
            NC = CN
            h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)

        return h
