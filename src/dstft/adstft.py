"""Module ADSTFT: Adaptive Differentiable Short-Time Fourier Transform.

This module provides the ADSTFT class, merged and modernized version.
"""

from math import pi

import torch
from torch import nn

from .base import BaseSTFT


class ADSTFT(BaseSTFT):
    """
    Adaptive Differentiable Short-Time Fourier Transform (ADSTFT) for PyTorch.

    This class implements an adaptive, differentiable version of the Short-Time Fourier Transform (STFT),
    where the window length and stride can be learned and can vary over time or frequency. It is designed
    for deep learning applications requiring flexible, trainable time-frequency representations.

    Args:
        x (torch.Tensor): Input signal tensor.
        win_length (float): Window length (can be adaptive).
        support (int): Support size (FFT size).
        stride (int): Stride size (hop length, can be adaptive).
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
        win_length (torch.Tensor): Adaptive window length parameter.
        support_size (int): FFT size.
        strides (torch.Tensor): Adaptive stride parameter.
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
        # --- Frames ---
        self.num_frames = int(
            1
            + torch.div(
                x.shape[-1] - (self.support_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
        )
        # --- Stride ---
        self.strides = nn.Parameter(
            torch.full(
                (1,), abs(stride), dtype=self.dtype, device=self.device
            ),
            requires_grad=self.stride_requires_grad,
        )
        # --- Window length ---
        self.win_length = nn.Parameter(
            torch.full(
                (1,),
                abs(win_length),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.win_requires_grad,
        )
        # --- Window power ---
        self.win_pow = nn.Parameter(
            torch.full(
                (1,),
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    @property
    def frames(self) -> torch.Tensor:
        expanded_stride = self.actual_strides.expand((self.num_frames,))
        frames = torch.zeros_like(expanded_stride)
        if self.first_frame:
            frames[0] = (
                self.actual_win_length.expand((
                    self.support_size,
                    self.num_frames,
                ))[:, 0].max(dim=0, keepdim=False)[0]
                - self.support_size
            ) / 2
        frames[1:] = frames[0] + expanded_stride[1:].cumsum(dim=0)
        return frames

    @property
    def effective_strides(self) -> torch.Tensor:
        expanded_stride = self.actual_strides.expand((self.num_frames,))
        effective_strides = torch.zeros_like(expanded_stride)
        effective_strides[1:] = expanded_stride[1:]
        cat = (
            torch.cat(
                (
                    torch.tensor(
                        [self.support_size],
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    self.actual_win_length.expand((
                        self.support_size,
                        self.num_frames,
                    )).max(dim=0, keepdim=False)[0],
                ),
                dim=0,
            ).diff()
            / 2
        )
        effective_strides = effective_strides - cat
        return effective_strides

    def stft(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        folded_x, idx_frac = self.unfold(x)
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac
        ).permute(2, 1, 0).unsqueeze(0)
        self.tapered_x = folded_x[:, :, None, :] * self.tap_win
        spectr = torch.fft.rfft(self.tapered_x)
        shift = torch.arange(
            end=self.num_frequencies,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        shift = idx_frac[:, None] * shift[None, :]
        shift = torch.exp(2j * pi * shift / self.support_size)[None, :, :, None]
        stft = spectr * shift
        # Select the first frequency bin from the last dimension and permute
        stft = stft[:, :, :, 0].permute(0, 2, 1)
        return stft

    def window_function(self, direction: str, idx_frac: torch.Tensor) -> torch.Tensor:
        """
        Generate the tapering window function or its derivative. Supports Hann/Hanning window.
        """
        base = torch.arange(
            0, self.support_size, 1, dtype=self.dtype, device=self.device
        )[:, None, None].expand([-1, self.num_frequencies, self.num_frames])
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
                f = torch.sin(
                    2
                    * pi
                    * (
                        base
                        + (self.actual_win_length - self.support_size + 1) / 2
                    )
                    / self.actual_win_length
                )
                d_tap_win = (
                    -pi
                    / self.actual_win_length.pow(2)
                    * ((self.support_size - 1) / 2 - base)
                    * f
                )
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                return d_tap_win

    def coverage(self) -> torch.Tensor:
        """
        Compute the coverage of the signal by the windows, in [0, 1].
        """
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

    def synt_win(self, direction: str, idx_frac: torch.Tensor) -> torch.Tensor:
        """Compute the synthesis window for overlap-add reconstruction (ADSTFT)."""
        # self.tap_win is (batch, num_frames, num_frequencies, support_size)

        # Get the analysis window (batch, num_frames, num_frequencies, support_size)
        analysis_window = self.tap_win

        wins = torch.zeros(
            self.num_frequencies, self.signal_length, device=self.device, dtype=self.dtype
        )

        for t in range(self.num_frames):
            start_idx = int(max(0, int(self.frames[t])))
            end_idx = int(min(self.signal_length, int(self.frames[t]) + self.support_size))
            start_dec = int(start_idx - int(self.frames[t]))
            end_dec = int(end_idx - int(self.frames[t]))

            length = min(end_idx - start_idx, self.support_size - start_dec)

            if length > 0:
                # Sum the squares of the analysis windows over the batch dimension, maintaining frequency dimension
                wins[:, start_idx:end_idx] += analysis_window[
                    0, t, :, start_dec : start_dec + length
                ].pow(2)

        self.wins = wins
        self.iwins = torch.zeros_like(wins)
        # Avoid division by zero
        self.iwins[wins > 0] = 1 / wins[wins > 0]

        # Now create the inverse tap window (itap_win) with shape (1, num_frames, num_frequencies, support_size)
        # The synthesis window is the analysis window divided by the sum of squared analysis windows
        itap_win = torch.zeros(
            (1, self.num_frames, self.num_frequencies, self.support_size),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(self.num_frames):
            start_idx = int(max(0, int(self.frames[t])))
            end_idx = int(min(self.signal_length, int(self.frames[t]) + self.support_size))
            start_dec = int(start_idx - int(self.frames[t]))
            end_dec = int(end_idx - int(self.frames[t]))

            length = min(end_idx - start_idx, self.support_size - start_dec)

            if length > 0:
                # Multiply each frequency band of the analysis window by the corresponding iwins value
                itap_win[0, t, :, start_dec : start_dec + length] = (
                    analysis_window[0, t, :, start_dec : start_dec + length]
                    * self.iwins[:, start_idx:end_idx]
                )

        return itap_win

    def idstft(self, stft: torch.Tensor) -> torch.Tensor:
        """Compute the inverse adaptive differentiable short-time Fourier transform (IADSTFT)."""
        # Inverse FFT along the frequency axis
        # stft is (batch, num_frequencies, num_frames)
        # Permute to (batch, num_frames, num_frequencies) for irfft
        # The output of irfft will be (batch, num_frames, num_frequencies, support_size)
        ifft = torch.fft.irfft(stft.permute(0, 2, 1).unsqueeze(-2), n=self.support_size, dim=-1)

        # Compute the synthesis window (adapted for ADSTFT shapes)
        self.itap_win = self.synt_win(None, None)

        # Apply synthesis window and sum over frequency dimension
        x_hat = (ifft * self.itap_win).sum(dim=-2)
        # Fold back to time domain
        x_hat = self.fold(x_hat)
        return x_hat
