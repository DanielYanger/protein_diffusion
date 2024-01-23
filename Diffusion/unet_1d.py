# Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py

import torch
from torch import nn
from functools import partial

import modules_1D
from modules_1D import default

class Unet1D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int = None, # type: ignore
        out_dim: int = None, # type: ignore
        dim_mults: tuple = (1, 2, 4, 8),
        channels: int = 3,
        self_condition: bool = False,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        attn_dim_head: int = 32,
        attn_heads: int = 4
    ):
        super().__init__()

        self.config = {
            'dim': dim,
            'init_dim': init_dim,
            'out_dim': out_dim,
            'dim_mults': dim_mults,
            'channels': channels,
            'self_condition': self_condition,
            'resnet_block_groups': resnet_block_groups,
            'learned_variance': learned_variance,
            'learned_sinusoidal_cond': learned_sinusoidal_cond,
            'random_fourier_features': random_fourier_features,
            'learned_sinusoidal_dim': learned_sinusoidal_dim,
            'sinusoidal_pos_emb_theta': sinusoidal_pos_emb_theta,
            'attn_dim_head': attn_dim_head,
            'attn_heads': attn_heads
        }

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(modules_1D.ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = modules_1D.RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = modules_1D.SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                modules_1D.Residual(modules_1D.PreNorm(dim_in, modules_1D.LinearAttention(dim_in))),
                modules_1D.Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = modules_1D.Residual(modules_1D.PreNorm(mid_dim, modules_1D.Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                modules_1D.Residual(modules_1D.PreNorm(dim_out, modules_1D.LinearAttention(dim_out))),
                modules_1D.Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1) # type: ignore

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs: # type: ignore
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups: # type: ignore
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    def save_unet(self, file_path):
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, str(file_path/f'unet.pt'))

    def load_unet(file_path):
        checkpoint = torch.load(str(file_path/f'unet.pt'))
        config = checkpoint['config']

        unet = Unet1D(
            config.dim,
            config.init_dim,
            config.out_dim,
            config.dim_mults,
            config.channels,
            config.self_condition,
            config.resnet_block_groups,
            config.learned_variance,
            config.learned_sinusoidal_cond,
            config.random_fourier_features,
            config.learned_sinusoidal_dim,
            config.sinusoidal_pos_emb_theta,
            config.attn_dim_head,
            config.attn_heads
        )

        unet.load_state_dict(checkpoint['state_dict'])
        return unet
