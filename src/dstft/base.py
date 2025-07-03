"""Base module for differentiable short-time Fourier transform (STFT) classes."""

from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn


class BaseSTFT(nn.Module):
    """Base class for short-time Fourier transform (STFT) modules.

    Args:
        x (torch.Tensor): Input signal tensor.
        win_length (float): Window length.
        support (int): Support size.
        stride (int): Stride size.
        magnitude_pow (float): Power for magnitude spectrogram.
        win_pow (float): Power for window function.
        win_requires_grad (bool): If True, window length is learnable.
        stride_requires_grad (bool): If True, stride is learnable.
        pow_requires_grad (bool): If True, window power is learnable.
        win_min (Optional[float]): Minimum window length.
        win_max (Optional[float]): Maximum window length.
        stride_min (Optional[float]): Minimum stride.
        stride_max (Optional[float]): Maximum stride.
        pow_min (Optional[float]): Minimum window power.
        pow_max (Optional[float]): Maximum window power.
        tapering_function (str): Window type ("hann" or "hanning").
        sr (int): Sample rate.
        window_transform (Optional[Callable]): Custom window transform.
        stride_transform (Optional[Callable]): Custom stride transform.
        dynamic_parameter (bool): If True, parameters can be dynamic.
        first_frame (bool): If True, special handling for first frame.
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
        win_min: Optional[float] = None,
        win_max: Optional[float] = None,
        stride_min: Optional[float] = None,
        stride_max: Optional[float] = None,
        pow_min: Optional[float] = None,
        pow_max: Optional[float] = None,
        tapering_function: str = "hann",
        sr: int = 16_000,
        window_transform: Optional[Callable] = None,
        stride_transform: Optional[Callable] = None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ) -> None:
        super().__init__()
        self.support_size: int = support
        self.num_frequencies: int = int(1 + self.support_size / 2)
        self.batch_size: int = x.shape[0]
        self.signal_length: int = x.shape[-1]
        self.device = x.device
        self.dtype = x.dtype
        self.win_requires_grad: bool = win_requires_grad
        self.stride_requires_grad: bool = stride_requires_grad
        self.pow_requires_grad: bool = pow_requires_grad
        self.tapering_function: str = tapering_function
        self.dynamic_parameter: bool = dynamic_parameter
        self.first_frame: bool = first_frame
        self.sr: int = sr
        self.magnitude_pow: float = magnitude_pow
        self.tap_win = None
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(self.dtype).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(self.dtype).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.win_min: float = win_min if win_min is not None else self.support_size / 20
        self.win_max: float = win_max if win_max is not None else self.support_size
        self.stride_min: float = stride_min if stride_min is not None else 0
        self.stride_max: float = stride_max if stride_max is not None else max(self.support_size, abs(stride))
        self.pow_min: float = pow_min if pow_min is not None else 0.001
        self.pow_max: float = pow_max if pow_max is not None else 1000
        self.window_transform: Callable = window_transform if window_transform is not None else self._window_transform
        self.stride_transform: Callable = stride_transform if stride_transform is not None else self._stride_transform

    def _clamp_parameter(self, value: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Clamp a parameter tensor between a minimum and maximum value."""
        return torch.minimum(
            torch.maximum(
                value,
                torch.full_like(value, min_val, dtype=self.dtype, device=self.device),
            ),
            torch.full_like(value, max_val, dtype=self.dtype, device=self.device),
        )

    def _window_transform(self, w_in: torch.Tensor) -> torch.Tensor:
        """Apply clamping constraints to the window length parameter."""
        return self._clamp_parameter(w_in, self.win_min, self.win_max)

    def _stride_transform(self, s_in: torch.Tensor) -> torch.Tensor:
        """Apply clamping constraints to the stride parameter."""
        return self._clamp_parameter(s_in, self.stride_min, self.stride_max)

    def _pow_transform(self, p_in: torch.Tensor) -> torch.Tensor:
        """Apply clamping constraints to the window power parameter."""
        return self._clamp_parameter(p_in, self.pow_min, self.pow_max)

    @property
    def actual_win_length(self) -> torch.Tensor:
        """Return the window length parameter after applying constraints."""
        return self.window_transform(self.win_length)

    @property
    def actual_strides(self) -> torch.Tensor:
        """Return the stride parameter after applying constraints."""
        return self.stride_transform(self.strides)

    @property
    def actual_pow(self) -> torch.Tensor:
        """Return the window power parameter after applying constraints."""
        return self._pow_transform(self.win_pow)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward STFT and return the magnitude spectrogram and complex STFT.

        Args:
            x (torch.Tensor): Input signal tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Magnitude spectrogram and complex STFT.
        """
        stft = self.stft(x, "forward")
        spec = (
            stft.abs().pow(self.magnitude_pow)[:, : self.num_frequencies]
            + torch.finfo(x.dtype).eps
        )
        return spec, stft

    def backward(self, x: torch.Tensor, dl_ds: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the loss w.r.t. window length parameter.

        Args:
            x (torch.Tensor): Input signal tensor.
            dl_ds (torch.Tensor): Gradient of the loss w.r.t. the spectrogram.
        Returns:
            torch.Tensor: Gradient w.r.t. window length parameter.
        """
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unfold the input signal into overlapping frames and compute the fractional part of the frame indices.

        Args:
            x (torch.Tensor): Input signal tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Folded signal and fractional frame indices.
        """
        idx_floor = self.frames.floor()
        idx_frac = self.frames - idx_floor
        idx_floor = idx_floor.long()[:, None].expand((
            self.num_frames,
            self.support_size,
        )) + torch.arange(0, self.support_size, device=self.device)
        idx_floor[idx_floor >= self.signal_length] = -1
        folded_x = x[:, idx_floor]
        folded_x[:, idx_floor < 0] = 0
        return folded_x, idx_frac

    def fold(self, folded_x: torch.Tensor) -> torch.Tensor:
        """Fold overlapping frames back into a single signal (overlap-add).

        Args:
            folded_x (torch.Tensor): Folded signal tensor.
        Returns:
            torch.Tensor: Reconstructed signal.
        """
        x_hat = torch.zeros(
            self.batch_size,
            self.signal_length,
            device=self.device,
            dtype=self.dtype,
        )
        for t in range(self.num_frames):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(
                self.signal_length - 1, int(self.frames[t]) + self.support_size
            )
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            x_hat[:, start_idx:end_idx] += folded_x[:, t, start_dec:end_dec]
        return x_hat

    def plot(
        self,
        spec: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        marklist: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (8, 5),
        f_hat: Optional[List[torch.Tensor]] = None,
        fs: Optional[int] = None,
        *,
        weights: bool = True,
        wins: bool = True,
        bar: bool = True,
        cmap: str = "magma",
        ylabel: str = "Frequency (Hz)",
        xlabel: str = "Time (frames)",
        interactive: bool = True,
        show_signal: bool = True,
    ) -> None:
        """Plot the spectrogram and related information interactively.

        Args:
            spec (torch.Tensor): Spectrogram tensor (batch, freq, time)
            x (Optional[torch.Tensor]): Original signal (optional)
            marklist (Optional[List[int]]): List of markers to add to the plot
            figsize (Tuple[int, int]): Figure size
            f_hat (Optional[List[torch.Tensor]]): List of frequency tracks to overlay
            fs (Optional[int]): Sampling rate (for frequency axis)
            weights (bool): Show window length distribution
            wins (bool): Show window shapes
            bar (bool): Show colorbar
            cmap (str): Colormap
            ylabel (str): Y-axis label
            xlabel (str): X-axis label
            interactive (bool): If True, use interactive widgets (Jupyter)
            show_signal (bool): If True, plot the original signal
        """
        try:
            from ipywidgets import IntSlider, interact
            has_widgets = interactive
        except ImportError:
            has_widgets = False
        f_max = spec.shape[-2] if fs is None else fs / 2

        def plot_spectrogram(frame=None):
            plt.figure(figsize=figsize)
            plt.title("Spectrogram")
            ax = plt.gca()
            im = ax.imshow(
                spec[0].detach().cpu().log(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=[0, spec.shape[-1], 0, f_max],
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar:
                plt.colorbar(im, ax=ax)
            if f_hat is not None:
                for f in f_hat:
                    plt.plot(f, linewidth=0.5, c="k", alpha=0.7)
            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, 1, c="gray")
            plt.tight_layout()
            plt.show()
        if has_widgets:
            interact(
                plot_spectrogram, frame=IntSlider(0, 0, spec.shape[-1] - 1)
            )
        else:
            plot_spectrogram()
        if weights:
            plt.figure(figsize=figsize)
            plt.title("Window Length Distribution")
            ax = plt.gca()
            display_win_length = self.actual_win_length
            if display_win_length.dim() == 0:
                display_win_length = display_win_length.unsqueeze(0)
            im = ax.imshow(
                display_win_length[: self.num_frequencies].detach().cpu(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            if bar:
                plt.colorbar(im, ax=ax)
                im.set_clim(self.win_min, self.win_max)
            plt.tight_layout()
            plt.show()
        if self.tap_win is not None and wins:
            plt.figure(figsize=figsize)
            plt.title("Window Shapes")
            ax = plt.gca()
            for i, start in enumerate(self.frames.detach().cpu()):
                ax.plot(
                    range(
                        int(start.floor().item()),
                        int(start.floor().item() + self.support_size),
                    ),
                    self.tap_win[:, i, :].squeeze().detach().cpu(),
                    c="#1f77b4",
                    alpha=0.5,
                )
            if marklist is not None:
                for elem in marklist:
                    plt.axvline(elem, 0, 1, c="gray")
            plt.tight_layout()
            plt.show()
        if show_signal and x is not None:
            plt.figure(figsize=figsize)
            plt.title("Original Signal")
            plt.plot(x.squeeze().cpu().numpy())
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()
