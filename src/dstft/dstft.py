"""Module DSTFT: Differentiable Short-Time Fourier Transform.

This module provides the DSTFT class, a modern and merged version.
"""

from math import pi

import torch
from torch import nn

from .base import BaseSTFT


class DSTFT(BaseSTFT):
    """
    Differentiable Short-Time Fourier Transform (STFT) for PyTorch.

    This class implements a differentiable version of the Short-Time Fourier Transform (STFT),
    allowing gradient-based learning of time-frequency representations. It supports custom window
    functions, learnable window length and stride, and is designed for integration with deep learning
    models in PyTorch.

    Args:
        x (torch.Tensor): Input signal tensor.
        win_length (float): Window length.
        support (int): Support size (FFT size).
        stride (int): Stride size (hop length).
        magnitude_pow (float, optional): Power for magnitude spectrogram. Defaults to 1.0.
        win_pow (float, optional): Power for window function. Defaults to 1.0.
        win_p (str, optional): Window parameterization type. Defaults to None.
        stride_p (str, optional): Stride parameterization type. Defaults to None.
        pow_p (str, optional): Power parameterization type. Defaults to None.
        win_requires_grad (bool, optional): If True, window length is learnable. Defaults to True.
        stride_requires_grad (bool, optional): If True, stride is learnable. Defaults to True.
        pow_requires_grad (bool, optional): If True, power is learnable. Defaults to False.
        win_min (float, optional): Minimum window length. Defaults to None.
        win_max (float, optional): Maximum window length. Defaults to None.
        stride_min (float, optional): Minimum stride. Defaults to None.
        stride_max (float, optional): Maximum stride. Defaults to None.
        pow_min (float, optional): Minimum power. Defaults to None.
        pow_max (float, optional): Maximum power. Defaults to None.
        tapering_function (str, optional): Window function type (e.g., 'hann'). Defaults to 'hann'.
        sr (int, optional): Sampling rate. Defaults to 16_000.
        window_transform (callable, optional): Custom window transform. Defaults to None.

    Attributes:
        win_length (torch.Tensor): Window length parameter.
        support_size (int): FFT size.
        strides (torch.Tensor): Stride parameter.
        ...
    """

    def __init__(
        self,
        x: torch.Tensor,
        win_length: float,
        support: int,
        stride: int,
        magnitude_pow: float = 1.0,
        win_pow: float = 1.0,
        win_requires_grad: bool = True,
        stride_requires_grad: bool = True,
        pow_requires_grad: bool = False,
        win_min: float | None = None,
        win_max: float | None = None,
        stride_min: float | None = None,
        stride_max: float | None = None,
        pow_min: float | None = None,
        pow_max: float | None = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform: callable = None,
        stride_transform: callable = None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ) -> None:
        super().__init__(
            x=x,
            win_length=win_length,
            support=support,
            stride=stride,
            magnitude_pow=magnitude_pow,
            win_pow=win_pow,
            win_requires_grad=win_requires_grad,
            stride_requires_grad=stride_requires_grad,
            pow_requires_grad=pow_requires_grad,
            win_min=win_min,
            win_max=win_max,
            stride_min=stride_min,
            stride_max=stride_max,
            pow_min=pow_min,
            pow_max=pow_max,
            tapering_function=tapering_function,
            sr=sr,
            window_transform=window_transform,
            stride_transform=stride_transform,
            dynamic_parameter=dynamic_parameter,
            first_frame=first_frame,
        )
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of {('hann', 'hanning')}, but got tapering_function={self.tapering_function}"
            )
        self.num_frames = 1 + int(
            torch.div(self.signal_length, stride, rounding_mode="floor")
        )
        # --- Stride ---
        stride_size = (1,)
        self.init_stride = abs(stride)
        self.strides = nn.Parameter(
            torch.full(
                stride_size,
                self.init_stride,
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.stride_requires_grad,
        )
        # --- Window length ---
        win_length_size = (1, 1)
        self.win_length = nn.Parameter(
            torch.full(
                win_length_size,
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )
        # --- Window power ---
        win_pow_size = (1, 1)
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    @property
    def frames(self) -> torch.Tensor:
        """Compute the temporal positions (indices) of the center of each frame."""
        expanded_stride = self.actual_strides.expand((self.num_frames,))
        frames = torch.zeros_like(expanded_stride)
        frames += expanded_stride.cumsum(dim=0)
        frames -= self.support_size / 2 + self.init_stride
        return frames

    @property
    def effective_strides(self) -> torch.Tensor:
        """Compute the strides between windows (not frames)."""
        expanded_stride = self.actual_strides.expand((self.num_frames,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = torch.cat(
            (
                torch.tensor(
                    [self.support_size], dtype=self.dtype, device=self.device
                ),
                self.actual_win_length.expand((
                    self.support_size,
                    self.num_frames,
                )).max(dim=0, keepdim=False)[0],
            ),
            dim=0,
        )
        offset = cat.diff() / 2
        effective_strides -= offset
        return effective_strides

    def stft(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        """Compute the Short-Time Fourier Transform or its derivative."""
        folded_x, idx_frac = self.unfold(x)
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac
        ).permute(1, 0).unsqueeze(0)
        self.tapered_x = folded_x * self.tap_win
        spectr = torch.fft.rfft(self.tapered_x)
        shift = torch.arange(
            end=self.num_frequencies,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]
        shift = torch.exp(2j * pi * shift / self.support_size)[None, ...]
        stft = spectr * shift
        return stft.permute(0, 2, 1)

    def inverse_dstft(self, stft: torch.Tensor) -> torch.Tensor:
        """Compute the inverse differentiable short-time Fourier transform (IDSTFT)."""
        ifft = torch.fft.irfft(stft, n=self.support_size, dim=-2)
        self.itap_win = self.synt_win(None, None)
        ifft = ifft.permute(0, -1, -2) * self.itap_win
        x_hat = self.fold(ifft)
        return x_hat

    def window_function(self, direction: str, idx_frac: torch.Tensor) -> torch.Tensor:
        """Generate the tapering window function or its derivative. Supports Hann/Hanning window."""
        base = torch.arange(
            0, self.support_size, 1, dtype=self.dtype, device=self.device
        )[:, None].expand([-1, self.num_frames])
        base = base - idx_frac
        mask1 = base.ge(
            torch.ceil((self.support_size - 1 + self.actual_win_length) / 2)
        )
        mask2 = base.le(
            torch.floor((self.support_size - 1 - self.actual_win_length) / 2)
        )
        if self.tapering_function in ("hann", "hanning"):
            if direction == "forward":
                self.tap_win = 0.5 - 0.5 * torch.cos(
                    2
                    * pi
                    * (
                        base
                        + (self.actual_win_length - self.support_size + 1) / 2
                    )
                    / self.actual_win_length
                )
                self.tap_win[mask1] = 0
                self.tap_win[mask2] = 0
                return self.tap_win.pow(self.win_pow)
            if direction == "backward":
                d_tap_win = -pi / self.actual_win_length.pow(2)
                d_tap_win = d_tap_win.expand_as(base)  # Ensure d_tap_win has the same shape as base
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.support_size * 2
                return d_tap_win

    def synt_win(self, direction: str, idx_frac: torch.Tensor) -> torch.Tensor:
        """Compute the synthesis window for overlap-add reconstruction."""
        wins = torch.zeros(self.signal_length, device=self.device, dtype=self.dtype)
        for t in range(self.num_frames):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(
                self.signal_length, int(self.frames[t]) + self.support_size
            )
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            wins[start_idx:end_idx] += (
                self.tap_win[:, t, start_dec:end_dec].squeeze().pow(2)
            )
        self.wins = wins
        self.iwins = torch.zeros(self.signal_length, device=self.device, dtype=self.dtype)
        self.iwins[self.wins > 0] = 1 / self.wins[self.wins > 0]
        itap_win = torch.zeros_like(self.tap_win)
        for t in range(self.num_frames):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(
                self.signal_length, int(self.frames[t]) + self.support_size
            )
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            itap_win[:, t, start_dec:end_dec] = self.tap_win[:, t, start_dec:end_dec] * self.iwins[start_idx:end_idx]
        return itap_win

    def coverage(self) -> torch.Tensor:
        """Compute the coverage of the signal by the windows, in [0, 1]."""
        covered_signal = torch.zeros(self.signal_length, dtype=torch.bool, device=self.device)
        for i in range(self.num_frames):
            current_win_length = self.actual_win_length.item()
            start_idx = int(self.frames[i] + self.support_size / 2 - current_win_length / 2)
            end_idx = int(self.frames[i] + self.support_size / 2 + current_win_length / 2)

            start_idx = max(0, start_idx)
            end_idx = min(self.signal_length, end_idx)

            if start_idx < end_idx:
                covered_signal[start_idx:end_idx] = True

        cov = covered_signal.sum().float() / self.signal_length
        return cov
