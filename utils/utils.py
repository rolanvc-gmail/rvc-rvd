import torch
import numpy as np
from inspect import isfunction


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    :param timesteps:
    :param s:
    :return:
    """
    steps = timesteps + 1
    x = np.linspace()
    alphas_cumprod = np.cos(((x/steps) + s) / (1+s) * np.py * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
