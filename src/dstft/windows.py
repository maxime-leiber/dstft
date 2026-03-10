"""Window functions for DSTFT.

This module will host differentiable analysis windows (Hann first) and
normalization strategies.
"""

from __future__ import annotations

from math import pi
from typing import Literal, TypeAlias

import torch


Normalization: TypeAlias = None | Literal["unit", "paper", "contract"]


def hann_window(
    *,
    n_fft: int,
    theta: torch.Tensor,
    idx_frac: torch.Tensor,
    freq_bins: int,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
    normalization: Normalization,
) -> torch.Tensor:
    """Hann window family on a fixed integer support.

    This function is intentionally designed as a template for future windows.

    Output shape:
        - scalar theta (FFT backend): [1, 1, n_fft]
        - time theta (FFT backend): [1, frames, n_fft]
        - frequency theta (DFT backend): [freq_bins, 1, n_fft]
        - time-frequency theta (DFT backend): [freq_bins, frames, n_fft]

    Notes:
        - `theta` is treated as a real-valued parameter (paper notation: θ).
        - `idx_frac` is used to shift the window continuously in time.
        - `normalization` is applied inside this function so that custom
          windows behave consistently.
    """
    theta_t = theta.to(device=device, dtype=dtype)

    # Canonical convention: theta is always 2D (freq_bins, frames) with
    # broadcastable singleton dimensions.
    if theta_t.ndim != 2:
        raise ValueError(
            "theta must be 2D with shape [freq_bins, frames] (broadcastable)"
        )

    f_dim, t_dim = theta_t.shape
    if f_dim not in {1, freq_bins}:
        raise ValueError("theta first dimension must be 1 or freq_bins")
    if t_dim not in {1, frames}:
        raise ValueError("theta second dimension must be 1 or frames")

    if t_dim != 1 and frames != idx_frac.numel():
        raise ValueError("frames must match idx_frac length for time-varying windows")

    theta_eval = theta_t
    theta_shape = (
        "scalar"
        if f_dim == 1 and t_dim == 1
        else "time"
        if f_dim == 1
        else "frequency"
        if t_dim == 1
        else "time-frequency"
    )

    # Paper-aligned centered parameterization:
    # window is centered at t_n and evaluated on a fixed discrete support.
    k_rel = torch.arange(n_fft, device=device, dtype=dtype) - (n_fft / 2)
    if theta_shape == "scalar":
        x = k_rel[None, None, :] - idx_frac.to(dtype)[None, :1, None]  # [1, 1, n_fft]
    elif theta_shape == "time":
        x = (
            k_rel[None, None, :] - idx_frac.to(dtype)[None, :, None]
        )  # [1, frames, n_fft]
    elif theta_shape == "frequency":
        x = (
            k_rel[None, None, :] - idx_frac.to(dtype)[None, :, None]
        )  # [1, frames, n_fft]
        x = x[:, :1, :].expand((freq_bins, 1, n_fft))
    else:
        x = (
            k_rel[None, None, :] - idx_frac.to(dtype)[None, :, None]
        )  # [1, frames, n_fft]
        x = x.expand((freq_bins, frames, n_fft))

    if normalization in {"paper", "contract"}:
        # Exact paper contraction: ω(x,θ) = (L/θ) ω_L((L/θ) x), where L=n_fft.
        scale = float(n_fft) / theta_eval
        if theta_shape in {"scalar", "time"}:
            x_eval = scale[..., None] * x
            u = x_eval + float(n_fft) / 2.0
            w = 0.5 - 0.5 * torch.cos(2.0 * pi * u / float(n_fft))
            w = w.masked_fill((u < 0.0) | (u >= float(n_fft)), 0.0)
            w = w * scale[..., None]
        else:
            x_eval = scale[..., None] * x
            u = x_eval + float(n_fft) / 2.0
            w = 0.5 - 0.5 * torch.cos(2.0 * pi * u / float(n_fft))
            w = w.masked_fill((u < 0.0) | (u >= float(n_fft)), 0.0)
            w = w * scale[..., None]
    else:
        # Direct centered Hann of effective length theta.
        u = x + theta_eval[..., None] / 2.0
        w = 0.5 - 0.5 * torch.cos(2.0 * pi * u / theta_eval[..., None])
        w = w.masked_fill((u < 0.0) | (u >= theta_eval[..., None]), 0.0)

    if normalization is None or normalization in {"paper", "contract"}:
        return w
    if normalization == "unit":
        denom = w.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        return w / denom
    raise ValueError(f"Unknown normalization: {normalization!r}")
