import torch as th
import torch.nn as nn

from .blocks import (
    SiLU,
    zero_module,
    normalization,
    timestep_embedding,
    TimestepEmbedSequential,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
)


class Network(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        conv_resample,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = hid_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hid_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, hid_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [hid_channels]
        ch = hid_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * hid_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * hid_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=hid_channels * mult,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = hid_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(hid_channels, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.hid_channels))

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)
