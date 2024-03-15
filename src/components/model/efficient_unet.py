import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import math
import random
import numpy as np
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.special import expm1

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from imagen_pytorch.t5 import get_encoded_dim, DEFAULT_T5_NAME

from src.components.utils import functions as func

# Setting reproducibility
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = func.beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = func.alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    def sample_random_times(self, batch_size, sample_interval ,*, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(*sample_interval)

    def get_condition(self, times):
        return func.maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device, t_min=0., t_max=1.):
        times = torch.linspace(t_max, t_min, int(t_max*self.num_timesteps) + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = func.default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(func.right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = func.log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = func.log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = func.log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = func.default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = func.right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  func.log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        noise = func.default(noise, lambda: torch.randn_like(x_from))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = func.right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  func.log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = func.right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  func.log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = func.right_pad_dims_to(x_t, log_snr)
        alpha, sigma = func.log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = func.right_pad_dims_to(x_t, log_snr)
        alpha, sigma = func.log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)

# norms and residuals
class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

ChanLayerNorm = partial(LayerNorm, dim = -3)

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

# attention pooling
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # qk rmsnorm

        q, k = map(func.l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if func.exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if func.exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = func.masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        return latents

# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if func.exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if func.exists(context):
            assert func.exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        q, k = map(func.l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # relative positional encoding (T5 style)

        if func.exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if func.exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# decoder
def Upsample(dim, dim_out = None):
    dim_out = func.default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = func.default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = func.default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if func.exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if func.exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if func.exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if func.exists(self.time_mlp) and func.exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if func.exists(self.cross_attn):
            assert func.exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context = cond) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift = scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = func.default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # cosine sim attention

        q, k = map(func.l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if func.exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if func.exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if func.exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if func.exists(context):
            assert func.exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias = False)
    )

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = func.default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = func.cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = func.default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [func.resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)

class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = True,
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        self_cond = False,
        resize_mode = 'nearest',
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
    ):
        super().__init__()

        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            func.print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # determine dimensions

        self.channels = channels
        self.channels_out = func.default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = func.default(init_dim, dim)

        self.self_cond = self_cond

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = func.default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert func.exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(dim = cond_dim, depth = 2, dim_head = attn_dim_head, heads = attn_heads, num_latents = attn_pool_num_latents) if attn_pool_text else None

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = func.cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = func.cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = func.cast_tuple(layer_attns, num_layers)
        layer_attns_depth = func.cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = func.cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = func.cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = func.cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass
        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)
        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_mid_attns_depth, **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = True) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += (channels if lowres_cond else 0)

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        func.zero_init_(self.final_conv)

        # resize mode

        self.resize_mode = resize_mode

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.func.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        self_cond = None,
        cond_drop_prob = 0.
    ):
        batch_size, device = x.shape[0], x.device

        # condition on self

        if self.self_cond:
            self_cond = func.default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not func.exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not func.exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        if func.exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # condition on input image

        assert not (self.has_cond_image ^ func.exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if func.exists(cond_images):
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            cond_images = func.resize_image_to(cond_images, x.shape[-1], mode = self.resize_mode)
            x = torch.cat((cond_images, x), dim = 1)

        # initial convolution

        x = self.init_conv(x)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)

        # text conditioning

        text_tokens = None

        if func.exists(text_embeds) and self.cond_on_text:

            # conditional dropout

            text_keep_mask = func.prob_mask_like((batch_size,), 1 - cond_drop_prob, device = device)

            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            # calculate text embeds

            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, :self.max_text_len]
            
            if func.exists(text_mask):
                text_mask = text_mask[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if func.exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            if func.exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens

            mean_pooled_text_tokens = text_tokens.mean(dim = -2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            null_text_hidden = self.null_text_hidden.to(t.dtype)

            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )
            
            t = t + text_hiddens

        # main conditioning tokens (c)

        c = time_tokens if not func.exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # initial resnet block (for memory efficient unet)

        if func.exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if func.exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if func.exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if func.exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if func.exists(self.final_res_block):
            x = self.final_res_block(x, t)

        if func.exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        return self.final_conv(x)

# null unet
class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = False,
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})
