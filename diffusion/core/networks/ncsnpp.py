import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
from .blocks import (
    get_timestep_embedding,
    get_sigmas,
    AttnBlockpp,
    ResBlockpp,
    get_act,
    conv3x3,
    default_initializer,
)


class Network(nn.Module):
    def __init__(
        self,
        nonlinearity,
        sigma_max,
        sigma_min,
        num_scales,
        nf,
        num_res_blocks,
        attn_resolutions,
        dropout,
        ch_mult,
        image_size,
        skip_rescale,
        init_scale,
        num_channels,
        scale_by_sigma,
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)
        self.register_buffer(
            "sigmas", torch.tensor(get_sigmas(sigma_max, sigma_min, num_scales))
        )

        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            image_size // (2**i) for i in range(num_resolutions)
        ]

        self.scale_by_sigma = scale_by_sigma
        self.skip_rescale = skip_rescale

        modules = []
        embed_dim = nf

        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        ResnetBlock = functools.partial(
            ResBlockpp,
            act=act,
            dropout=dropout,
            init_scale=init_scale,
            skip_rescale=skip_rescale,
            temb_dim=nf * 4,
        )

        channels = num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch))

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True))

        modules.append(
            nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        )
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
        modules = self.all_modules
        m_idx = 0
        timesteps = time_cond
        used_sigmas = self.sigmas[time_cond.long()]
        temb = get_timestep_embedding(timesteps, self.nf)

        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h
