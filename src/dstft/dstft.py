"""Module DSTFT: Differentiable Short-Time Fourier Transform.

This module provides the DSTFT class, a modern and merged version.
"""

from math import pi

import matplotlib.pyplot as plt
import torch
from torch import nn


class DSTFT(nn.Module):
    """Differentiable short-time Fourier transform (DSTFT) module.
    Modern, merged version.
    """

    def __init__(
        self,
        x: torch.Tensor,
        win_length: float,
        support: int,
        stride: int,
        magnitude_pow: float = 1.0,
        win_pow: float = 1.0,
        win_p: str | None = None,
        stride_p: str | None = None,
        pow_p: str | None = None,
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
        window_transform=None,
        stride_transform=None,
        dynamic_parameter: bool = False,
        first_frame: bool = False,
    ):
        super().__init__()
        # --- Main attributes initialization (see main version) ---
        self.support_size = support
        self.num_frequencies = int(1 + self.support_size / 2)
        self.batch_size = x.shape[0]
        self.signal_length = x.shape[-1]
        self.device = x.device
        self.dtype = x.dtype
        self.num_frames = 1 + int(
            torch.div(self.signal_length, stride, rounding_mode="floor")
        )
        self.win_requires_grad = win_requires_grad
        self.stride_requires_grad = stride_requires_grad
        self.pow_requires_grad = pow_requires_grad
        self.tapering_function = tapering_function
        self.dynamic_parameter = dynamic_parameter
        self.first_frame = first_frame
        self.sr = sr
        self.magnitude_pow = magnitude_pow
        self.tap_win = None
        self.register_buffer(
            "eps",
            torch.tensor(
                torch.finfo(torch.float).eps,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "min",
            torch.tensor(
                torch.finfo(torch.float).min,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        # --- Stride ---
        self.stride_min = stride_min if stride_min is not None else 0
        self.stride_max = (
            stride_max
            if stride_max is not None
            else max(self.support_size, abs(stride))
        )
        self.stride_transform = (
            stride_transform
            if stride_transform is not None
            else self.__stride_transform
        )
        if stride_p is None:
            stride_size = (1,)
        elif stride_p == "t":
            stride_size = (self.num_frames,)
        else:
            raise ValueError(
                f"stride_p error {stride_p}. Must be 't' or None."
            )
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
        self.win_min = (
            win_min if win_min is not None else self.support_size / 20
        )
        self.win_max = win_max if win_max is not None else self.support_size
        self.window_transform = (
            window_transform
            if window_transform is not None
            else self.__window_transform
        )
        if win_p is None:
            win_length_size = (1, 1)
        elif win_p == "t":
            win_length_size = (1, self.num_frames)
        else:
            raise ValueError(
                f"Invalid win_p value: {win_p}. Must be 't' or None. Frequency-varying windows might require a different module like ADSTFT."
            )
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
        self.pow_min = pow_min if pow_min is not None else 0.001
        self.pow_max = pow_max if pow_max is not None else 1000
        if pow_p is None:
            win_pow_size = (1, 1)
        elif pow_p == "t":
            win_pow_size = (1, self.num_frames)
        else:
            raise ValueError(
                f"Invalid pow_p value: {pow_p}. Must be 't' or None."
            )
        self.win_pow = nn.Parameter(
            torch.full(
                win_pow_size,
                abs(win_pow),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=self.pow_requires_grad,
        )

    def _clamp_parameter(
        self, value: torch.Tensor, min_val: float, max_val: float
    ) -> torch.Tensor:
        """Clamp a parameter tensor between a minimum and maximum value."""
        return torch.minimum(
            torch.maximum(
                value,
                torch.full_like(
                    value, min_val, dtype=self.dtype, device=self.device
                ),
            ),
            torch.full_like(
                value, max_val, dtype=self.dtype, device=self.device
            ),
        )

    def __window_transform(self, w_in: torch.Tensor) -> torch.Tensor:
        """Apply clamping constraints to the window length parameter."""
        return self._clamp_parameter(w_in, self.win_min, self.win_max)

    def __stride_transform(self, s_in: torch.Tensor) -> torch.Tensor:
        """Apply clamping constraints to the stride parameter."""
        return self._clamp_parameter(s_in, self.stride_min, self.stride_max)

    def __pow_transform(self, p_in: torch.Tensor) -> torch.Tensor:
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
        return self.__pow_transform(self.win_pow)

    @property
    def frames(self) -> torch.Tensor:
        """Compute the temporal positions (indices) of the center of each frame."""
        expanded_stride = self.actual_strides.expand((self.num_frames,))
        frames = torch.zeros_like(expanded_stride)
        frames += expanded_stride.cumsum(dim=0)
        frames -= self.support_size / 2 + self.init_stride
        return frames

    @property
    def effective_strides(self):
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

    def forward(self, x: torch.Tensor) -> tuple:
        """Compute the forward STFT and return the magnitude spectrogram and complex STFT."""
        stft = self.stft(x, "forward")
        spec = (
            stft.abs().pow(self.magnitude_pow)[:, : self.num_frequencies]
            + torch.finfo(x.dtype).eps
        )
        return spec, stft

    def backward(self, x: torch.Tensor, dl_ds: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the loss w.r.t. window length parameter."""
        dstft_dp = self.stft(x, "backward")
        dl_dp = torch.conj(dl_ds) * dstft_dp
        dl_dp = dl_dp.sum().real.expand(self.win_length.shape)
        return dl_dp

    def stft(self, x: torch.Tensor, direction: str):
        """Compute the Short-Time Fourier Transform or its derivative."""
        folded_x, idx_frac = self.unfold(x)
        self.tap_win = self.window_function(
            direction=direction, idx_frac=idx_frac
        ).permute(1, 0)
        self.folded_x = folded_x[:, :, :]
        self.tap_win = self.tap_win[None, :, :]
        self.num_framesapered_x = self.folded_x * self.tap_win
        spectr = torch.fft.rfft(self.num_framesapered_x)
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

    def unfold(self, x: torch.Tensor) -> tuple:
        """Unfold the input signal into overlapping frames and compute the fractional part of the frame indices."""
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
        """Fold overlapping frames back into a single signal (overlap-add)."""
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

    def window_function(self, direction: str, idx_frac) -> torch.Tensor:
        """Generate the tapering window function or its derivative. Supports Hann/Hanning window."""
        if self.tapering_function not in {"hann", "hanning"}:
            raise ValueError(
                f"tapering_function must be one of '{('hann', 'hanning')}', but got padding_mode='{self.tapering_function}'"
            )
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
                d_tap_win[mask1] = 0
                d_tap_win[mask2] = 0
                d_tap_win = d_tap_win / self.support_size * 2
                return d_tap_win
        return None

    def synt_win(self, direction: str, idx_frac) -> torch.Tensor:
        """Compute the synthesis window for overlap-add reconstruction."""
        wins = torch.zeros(self.signal_length)
        for t in range(self.num_frames):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(
                self.signal_length, int(self.frames[t]) + self.support_size
            )
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            wins[start_idx:end_idx] += (
                self.tap_win[:, t, start_dec:end_dec].squeeze().detach().cpu()
            )
        self.wins = wins
        self.iwins = torch.zeros(self.signal_length)
        self.iwins[self.wins > 0] = 1 / self.wins[self.wins > 0]
        itap_win = torch.zeros_like(self.tap_win)
        for t in range(self.num_frames):
            start_idx = max(0, int(self.frames[t]))
            end_idx = min(
                self.signal_length, int(self.frames[t]) + self.support_size
            )
            start_dec = start_idx - int(self.frames[t])
            end_dec = end_idx - int(self.frames[t])
            itap_win[:, t, start_dec:end_dec] = self.iwins[start_idx:end_idx]
        return itap_win

    def coverage(self):
        """Compute the coverage of the signal by the windows, in [0, 1]."""
        expanded_win, _ = self.actual_win_length.expand((
            self.support_size,
            self.num_frames,
        )).min(dim=0, keepdim=False)
        cov = expanded_win[0]
        maxi = self.frames[0] + self.support_size / 2 + expanded_win[0] / 2
        for i in range(1, self.num_frames):
            start = torch.min(
                self.signal_length * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i]
                    + self.support_size / 2
                    - expanded_win[i] / 2,
                ),
            )
            end = torch.min(
                self.signal_length * torch.ones_like(expanded_win[i]),
                torch.max(
                    torch.zeros_like(expanded_win[i]),
                    self.frames[i]
                    + self.support_size / 2
                    + expanded_win[i] / 2,
                ),
            )
            if end > maxi:
                cov += end - torch.max(start, maxi)
                maxi = end
        cov /= self.signal_length
        return cov

    def plot(
        self,
        spec: torch.Tensor,
        x: torch.Tensor = None,
        marklist=None,
        figsize=(8, 5),
        f_hat=None,
        fs=None,
        *,
        weights=True,
        wins=True,
        bar=True,
        cmap="magma",
        ylabel="Frequency (Hz)",
        xlabel="Time (frames)",
        interactive=True,
        show_signal=True,
    ):
        """
        Plot the spectrogram and related information interactively.

        Args:
            spec: Spectrogram tensor (batch, freq, time)
            x: Original signal (optional)
            marklist: List of markers to add to the plot
            figsize: Figure size
            f_hat: List of frequency tracks to overlay
            fs: Sampling rate (for frequency axis)
            weights: Show window length distribution
            wins: Show window shapes
            bar: Show colorbar
            cmap: Colormap
            ylabel: Y-axis label
            xlabel: X-axis label
            interactive: If True, use interactive widgets (Jupyter)
            show_signal: If True, plot the original signal
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
            im = ax.imshow(
                self.actual_win_length[: self.num_frequencies].detach().cpu(),
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
