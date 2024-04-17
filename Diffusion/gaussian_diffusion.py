# From https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py

import math
import json
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from Diffusion.unet_1d import Unet1D
from Diffusion.modules_1D import default, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, identity, create_folder, left_broadcast

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model: Unet1D,
        *,
        seq_length = 1000,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()


        self.config = {
            'seq_length': seq_length,
            'timesteps': timesteps,
            'sampling_timesteps': sampling_timesteps,
            'objective': objective,
            'beta_schedule': beta_schedule,
            'ddim_sampling_eta': ddim_sampling_eta,
            'auto_normalize': auto_normalize
        }

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    def ddim_single_step(self, batch_size, img, time, prev_sample, eta, clip_denoised = True):
        shape = (batch_size, self.channels, self.seq_length)
        device = self.betas.device
        time_next = time - 1
        
        # time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
        self_cond = None
        pred_noise, x_start, *_ = self.model_predictions(img, time, self_cond, clip_x_start = clip_denoised)
        
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        sigma = sigma.view(batch_size, 1, 1)

        img_mean = x_start * alpha_next.view(batch_size, 1, 1).sqrt() + c.view(batch_size, 1, 1)
        log_prob = (
            -((prev_sample.detach() - img_mean) ** 2) / (2 * (sigma**2)) # type: ignore
            - torch.log(sigma)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, batch_size = 0, eta=None):
        if eta is None:
            eta = self.ddim_sampling_eta

        if shape is None:
            shape = (batch_size, self.channels, self.seq_length)

        batch, device, total_timesteps, sampling_timesteps, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device = device)

        x_start = None

        log_probs = []
        all_latents = [img]
        times = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue
            
            times.append(time)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img_mean = x_start * alpha_next.sqrt() + c * pred_noise
            
            img = img_mean + sigma * noise

            all_latents.append(img)
            log_prob = (
                -((img.detach() - img_mean) ** 2) / (2 * (sigma**2)) # type: ignore
                - torch.log(sigma)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

            log_probs.append(log_prob.mean(dim=tuple(range(1, log_prob.ndim))))


        img = self.unnormalize(img)
        return img, all_latents, log_probs, torch.tensor(times, device = device)
 
    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        sample_fn = self.ddim_sample
        img, *_ = sample_fn((batch_size, channels, seq_length))
        return img

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
    
    def save_model(self, file_path, milestone = ''):
        create_folder(file_path)
        self.model.save_unet(file_path, milestone=milestone)

        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
        }

        torch.save(checkpoint, str(f'{file_path}/diffusion-model{milestone}.pt'))

    def load_diffusion(file_path, device):

        unet = Unet1D.load_unet(file_path)
        
        checkpoint = torch.load(str(f'{file_path}/diffusion-model.pt'), map_location=device)
        config = checkpoint['config']

        diffusion = GaussianDiffusion1D(
            unet,
            seq_length=config["seq_length"],
            timesteps=config["timesteps"],
            sampling_timesteps=config["sampling_timesteps"],
            objective=config["objective"],
            beta_schedule=config["beta_schedule"],
            ddim_sampling_eta=config["ddim_sampling_eta"],
            auto_normalize=config["auto_normalize"]
        )

        diffusion.load_state_dict(checkpoint['state_dict'])

        return diffusion


    # based on https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py#L39 
    def ddim_step_log_prob(self, 
                           model_output: torch.FloatTensor, 
                           timestep, 
                           sample: torch.FloatTensor,
                           eta: float = 0.0,
                           use_clipped_model_output: bool = False,
                           prev_sample: Optional[torch.FloatTensor] = None):
        prev_timestep = timestep - self.num_timesteps // self.sampling_timesteps
        prev_timestep = torch.clamp(prev_timestep, 0, self.num_timesteps - 1) # type: ignore
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if use_clipped_model_output else identity
        
        alpha = self.alphas_cumprod[timestep] # type: ignore
        alpha = left_broadcast(alpha, sample.shape).to(sample.device)

        alpha_prev = self.alphas_cumprod[prev_timestep] # type: ignore
        alpha_prev = left_broadcast(alpha_prev, sample.shape).to(sample.device)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(sample, timestep, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(sample, timestep, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(sample, timestep, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(sample, timestep, x_start)
        else:
            raise Exception("Invalid objective")
        
        if use_clipped_model_output:
            pred_noise = self.predict_noise_from_start(sample, timestep, x_start)

        std = eta * ((1 - alpha / alpha_prev) * (1 - alpha_prev) / (1 - alpha)).sqrt()
        pred_sample_direction = (1 - alpha_prev - std**2) ** (0.5) * pred_noise
        prev_sample_mean = alpha_prev ** (0.5) * x_start + pred_sample_direction

        if prev_sample is None:
            variance_noise = torch.randn_like(model_output)
            prev_sample = prev_sample_mean + std * variance_noise # type: ignore

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std**2)) # type: ignore
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample.type(sample.dtype), log_prob # type: ignore
    
    # based on https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/pipeline_with_logprob.py
    @torch.no_grad()
    def pipeline_with_logprob(self,
                              eta: float = 0.0,
                              batch_size: int = 16):
        
        inference_steps = self.sampling_timesteps
        total_steps = self.num_timesteps
        times = torch.linspace(-1, total_steps - 1, steps=inference_steps + 1)  
        times = torch.tensor(list(reversed(times.int().tolist()))[:-1], device = self.betas.device)

        shape = (batch_size, self.channels, self.seq_length)
        latents = torch.randn(shape, device = self.betas.device) # type: ignore

        all_latents = [latents]
        all_log_probs = []

        for timestep in tqdm(times, desc = "Sampling with Log Prob Loop Time Step"):
            # No CFG because that's only for conditional distributions
            time = torch.full((batch_size,), timestep, device=self.betas.device, dtype=torch.long)
            latent_model_input = latents
            model_output = self.model(latent_model_input, time)

            latents, log_prob = self.ddim_step_log_prob(model_output, time, latents, eta) # type: ignore

            all_latents.append(latents)
            all_log_probs.append(log_prob)

        return latents, all_latents, all_log_probs, times