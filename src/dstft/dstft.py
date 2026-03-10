"""Public DSTFT module.

This file contains the single user-facing `DSTFT` class.

Design principles:
- `DSTFT` is an API/state holder (parameters, modes, validation).
- Mathematical kernels are implemented in `dstft/_core.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias

import torch
from torch import nn

from . import _core, windows

Normalization: TypeAlias = None | Literal["unit", "paper", "contract"]
WindowMode: TypeAlias = Literal[
    "fixed",
    "constant",
    "time",
    "frequency",
    "time-frequency",
]
HopMode: TypeAlias = Literal["fixed", "constant", "time"]


class WindowFn(Protocol):
    def __call__(
        self,
        *,
        n_fft: int,
        theta: torch.Tensor,
        idx_frac: torch.Tensor,
        freq_bins: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
        normalization: Normalization,
    ) -> torch.Tensor: ...


WindowSpec: TypeAlias = Literal["hann"] | WindowFn


class DSTFT(nn.Module):
    """Differentiable Short-Time Fourier Transform.

    This package exposes a single user-facing class which internally dispatches to:

    - an FFT backend when the window varies at most over time
    - a DFT backend when the window varies over frequency or time-frequency

    The backend choice is *not* exposed as a user parameter.

    Notes:
        This implementation currently supports real-valued 1D signals with shape
        ``[batch, time]``.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        win_length: float | int | None = None,
        hop_length: float | int | None = None,
        window_mode: WindowMode = "time",
        hop_mode: HopMode = "fixed",
        window: WindowSpec = "hann",
        normalization: Normalization = None,
        hop_length_min: float = 1.0,
        hop_length_max: float | None = None,
        win_length_min: float | None = None,
        win_length_max: float | None = None,
        magnitude_power: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        """Initialize a DSTFT module.

        Args:
            n_fft: FFT size (frame length in samples).
            win_length: Initial window length (in samples). If ``None``, defaults
                to ``n_fft``.
            hop_length: Initial hop length (in samples). If ``None``, defaults to
                ``n_fft // 4``.
            window_mode: Window parameterization mode.
            hop_mode: Hop parameterization mode.
            window: Window specification. Either a string identifier (currently
                only ``"hann"``) or a callable implementing ``WindowFn``.
            normalization: Window normalization mode passed to the window
                function.
            hop_length_min: Lower bound for hop length.
            hop_length_max: Upper bound for hop length. If ``None``, defaults to
                ``n_fft``.
            win_length_min: Lower bound for window length.
            win_length_max: Upper bound for window length. If ``None``, defaults
                to ``n_fft``.
            magnitude_power: Exponent applied to the magnitude spectrogram.
            eps: Small positive value for numerical stability.

        Raises:
            ValueError: If a parameter is out of range.
        """
        super().__init__()

        if n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {n_fft}")
        if hop_length is not None and float(hop_length) <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if win_length is not None and win_length <= 0:
            raise ValueError(
                f"win_length must be positive when provided, got {win_length}"
            )
        if magnitude_power <= 0:
            raise ValueError(f"magnitude_power must be positive, got {magnitude_power}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.n_fft = int(n_fft)
        self.freq_bins = self.n_fft // 2 + 1
        self.window = window
        self.window_mode = window_mode
        self.hop_mode = hop_mode
        self.normalization = normalization
        self.hop_length_min = float(hop_length_min)
        self.hop_length_max = (
            float(hop_length_max) if hop_length_max is not None else float(self.n_fft)
        )
        self.win_length_min = float(
            win_length_min if win_length_min is not None else max(1, self.n_fft // 100)
        )
        self.win_length_max = (
            float(win_length_max) if win_length_max is not None else float(self.n_fft)
        )
        self.magnitude_power = float(magnitude_power)
        self.eps = float(eps)

        initial_win_length = (
            float(win_length) if win_length is not None else float(n_fft)
        )
        initial_hop_length = (
            float(hop_length) if hop_length is not None else float(self.n_fft // 4)
        )

        # Default lower bound for window length.
        # Policy: if the user does not provide `win_length_min`, we use the
        # constructor hop length (a constant) as the default `win_min`.
        self._default_win_length_min = (
            float(win_length_min)
            if win_length_min is not None
            else float(initial_hop_length)
        )

        self._initialized = False
        self._num_frames: int | None = None
        self._init_signal_length: int | None = None
        self._init_device: torch.device | None = None
        self._init_dtype: torch.dtype | None = None

        # Internal learnable parameters are stored in unconstrained form
        # (`raw_*`). Public accessors (`win_length`, `hop_length`) return the
        # effective constrained values.
        self._raw_win_length: torch.Tensor
        self._raw_hop_length: torch.Tensor

        # Cached for inspection/debugging (set during forward).
        self.analysis_window: torch.Tensor | None = None

        # Important: frame-shaped parameters depend on the input signal length.
        # They are allocated in `initialize(x)` (not in `forward()`).
        init_win = torch.tensor([[initial_win_length]])
        # Initialize raw so that sigmoid(raw) yields the requested initial value.
        win_min_init = float(self._default_win_length_min)
        win_p = (initial_win_length - win_min_init) / (float(self.n_fft) - win_min_init)
        win_p = float(max(1e-4, min(1.0 - 1e-4, win_p)))
        init_raw_win = torch.log(torch.tensor(win_p)) - torch.log1p(
            torch.tensor(-win_p)
        )
        self._raw_win_length = nn.Parameter(
            init_raw_win.expand_as(init_win).clone(),
            requires_grad=(self.window_mode != "fixed"),
        )

        init_hop = torch.tensor([initial_hop_length])
        hop_p = (initial_hop_length - float(self.hop_length_min)) / (
            float(self.hop_length_max) - float(self.hop_length_min)
        )
        hop_p = float(max(1e-4, min(1.0 - 1e-4, hop_p)))
        init_raw_hop = torch.log(torch.tensor(hop_p)) - torch.log1p(
            torch.tensor(-hop_p)
        )
        self._raw_hop_length = nn.Parameter(
            init_raw_hop.expand_as(init_hop).clone(),
            requires_grad=(self.hop_mode != "fixed"),
        )

        # Pre-bind the window callable once (no branching in forward).
        self._window_fn: WindowFn = self._resolve_window(window)

        # Pre-bind the backend transform once (no branching in forward).
        # For now, only FFT is implemented. Later, DFT is selected when
        # window_mode implies frequency dependence.
        self._transform_fn: Callable[..., torch.Tensor]
        if window_mode in {"frequency", "time-frequency"}:
            # DFT backend required for frequency-dependent windows.
            self._transform_fn = _core.adstft_dft_forward
        else:
            self._transform_fn = _core.fdstft_fft_forward

        # Parameters (introduced incrementally in later sprints)
        # - window length parameter(s): θ, θ_n, θ_m, θ_{m,n}
        # - hop / frame position parameter(s)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the DSTFT.

        Args:
            x: Real-valued signal tensor of shape `[batch, time]`.

        Returns:
            A tuple ``(spec, stft)`` where:

            - ``spec`` is the magnitude spectrogram of shape ``[batch, freq, frames]``.
            - ``stft`` is the complex-valued DSTFT of shape ``[batch, freq, frames]``.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"x must have shape [batch, time], got {x.shape}")
        if not self._initialized:
            raise RuntimeError(
                "DSTFT must be initialized with `initialize(x)` before the first "
                "forward call. This allocates input-dependent parameters and caches "
                "the number of frames."
            )

        # 1) Frame center positions (learnable or fixed)
        frame_positions = self.frame_centers(device=x.device, dtype=x.dtype)

        # 2) Unfold via floor/frac decomposition (paper Eq. 25 logic)
        frames, idx_floor, idx_frac = _core.unfold_floor_frac(
            x=x,
            frame_positions=frame_positions,
            n_fft=self.n_fft,
        )

        # 3) Analysis window generation
        freq_bins = self.freq_bins
        theta = self._effective_win_length(device=x.device, dtype=x.dtype)

        # When the window is fixed, avoid constructing a per-frame window.
        idx_frac_for_window = idx_frac if self.window_mode != "fixed" else idx_frac[:1]
        analysis_window = self._window_fn(
            n_fft=self.n_fft,
            theta=theta,
            idx_frac=idx_frac_for_window,
            freq_bins=freq_bins,
            frames=frame_positions.numel(),
            device=x.device,
            dtype=x.dtype,
            normalization=self.normalization,
        )

        if self.window_mode in {"fixed", "constant", "time"}:
            # FFT backend accepts either [frames, n_fft] or broadcastable [1, frames, n_fft].
            if analysis_window.ndim == 3:
                if analysis_window.shape[0] != 1:
                    raise RuntimeError(
                        "FFT backend requires analysis_window with leading dimension 1 when 3D"
                    )
            elif analysis_window.ndim != 2:
                raise RuntimeError(
                    "FFT backend requires analysis_window with 2D or 3D shape"
                )

        if (
            self.window_mode in {"frequency", "time-frequency"}
            and analysis_window.ndim != 3
        ):
            raise RuntimeError(
                "DFT backend requires analysis_window with shape [freq_bins, frames, n_fft] "
                "(with possible broadcastable freq_bins=1 or frames=1)"
            )

        # Expose last analysis window for debugging/visualization.
        self.analysis_window = analysis_window

        # 4) Backend dispatch (FFT vs DFT)
        # In early MVP sprints, we start with FFT-only (window_mode None/constant/time).
        stft = self._transform_fn(
            frames=frames,
            analysis_window=analysis_window,
            n_fft=self.n_fft,
            idx_frac=idx_frac,
            idx_floor=idx_floor,
        )

        # 5) Spectrogram magnitude
        spec = stft.abs().pow(self.magnitude_power) + self.eps
        return spec, stft

    def inverse(
        self,
        stft: torch.Tensor,
        *,
        method: str = "auto",
        cg_max_iter: int = 20,
        cg_tol: float = 1e-8,
        cg_lambda: float = 1e-6,
    ) -> torch.Tensor:
        """Inverse DSTFT.

        This reconstructs a real-valued signal using synthesis=analysis window
        and a WOLA normalization:

            x_hat = num / (den + eps)

        Args:
            stft: Complex STFT tensor of shape `[batch, freq, frames]`.
            method: Inversion method.
                - ``"auto"``: Uses ``"wola"`` for FFT backend and ``"wola"`` for
                  DFT backend.
                - ``"wola"``: Fast approximate inverse based on the adjoint and
                  pointwise energy normalization.
                - ``"cg"``: Approximate least-squares inverse (DFT backend) by
                  solving ``(A* A + λI)x = A* s`` with preconditioned conjugate
                  gradients.
            cg_max_iter: Maximum number of CG iterations (DFT backend only).
            cg_tol: Relative residual tolerance for CG (DFT backend only).
            cg_lambda: Tikhonov regularization ``λ`` (DFT backend only).

        Returns:
            Reconstructed signal tensor of shape `[batch, time]`.

        Raises:
            TypeError: If ``stft`` is not a tensor.
            ValueError: If ``stft`` does not have shape ``[batch, freq, frames]`` or
                the frequency dimension does not match ``n_fft``.
            RuntimeError: If the module has not been initialized.
            ValueError: If ``method`` is unknown.

        Notes:
            Under the framing convention (frame centers with implicit
            zero-padding), exact reconstruction is expected on the covered
            region (typically up to `signal_length - n_fft//2`). The final
            samples may be uncovered depending on the hop policy.

            **Perfect reconstruction (FFT backend).**
            The forward computes per-frame windowed time-domain frames
            ``y_n[k] = x[i_n + k] * w_n[k]`` with ``i_n = floor(t_n)``, followed
            by an FFT over the *local* index ``k`` and the article Eq. (25) phase
            factor ``exp(-j 2π i_n m / n_fft)``. The inverse first cancels that
            Eq. (25) factor, applies an ``irfft`` to recover ``y_n[k]``, and
            then overlap-adds using the pointwise-normalized WOLA rule:

            ``x_hat[p] = (sum_n y_n[p - i_n] * w_n[p - i_n]) / (sum_n w_n[p - i_n]^2 + eps)``.

            Since the numerator equals ``x[p] * sum_n w_n[p - i_n]^2``, this
            yields ``x_hat[p] = x[p]`` for every sample ``p`` such that the
            denominator is non-zero (i.e. the sample is covered by at least one
            frame).

            **DFT backend (frequency-dependent windows).**
            When the window varies across frequency bins (``window_mode`` is
            ``"frequency"`` or ``"time-frequency"``), this method uses the
            adjoint operator ``A*`` followed by a pointwise energy
            normalization. This corresponds to a fast diagonal approximation of
            ``(A* A)^{-1}`` and is generally not exact.
        """
        if not isinstance(stft, torch.Tensor):
            raise TypeError(f"stft must be a torch.Tensor, got {type(stft)}")
        if stft.ndim != 3:
            raise ValueError(
                f"stft must have shape [batch, freq, frames], got {stft.shape}"
            )
        if not self._initialized:
            raise RuntimeError("DSTFT must be initialized before inverse")
        if self._init_signal_length is None:
            raise RuntimeError("DSTFT initialization metadata is missing")
        method = method.lower()
        if method == "auto":
            method = (
                "wola"
                if self.window_mode not in {"frequency", "time-frequency"}
                else "cg"
            )
        if method not in {"wola", "cg"}:
            raise ValueError(f"Unknown inverse method: {method!r}")

        expected_freq = self.freq_bins
        if stft.shape[1] != expected_freq:
            raise ValueError(
                f"stft must have freq dimension {expected_freq}, got {stft.shape[1]}"
            )

        # Keep inverse dtype/device consistent with the forward pass.
        inv_dtype = stft.real.dtype
        frame_positions = self.frame_centers(device=stft.device, dtype=inv_dtype)
        theta = self._effective_win_length(device=stft.device, dtype=inv_dtype)

        # Recompute idx_frac to generate the same analysis/synthesis window.
        idx_floor = frame_positions.floor().to(torch.int64)
        idx_frac = (frame_positions - idx_floor.to(frame_positions.dtype)).to(inv_dtype)
        analysis_window = self._window_fn(
            n_fft=self.n_fft,
            theta=theta,
            idx_frac=idx_frac,
            freq_bins=self.freq_bins,
            frames=frame_positions.numel(),
            device=stft.device,
            dtype=inv_dtype,
            normalization=self.normalization,
        )

        if self.window_mode in {"frequency", "time-frequency"}:
            # DFT backend: use adjoint + diagonal (WOLA) normalization.
            if analysis_window.ndim != 3:
                raise RuntimeError(
                    "DFT inverse requires analysis_window with shape [freq_bins, frames, n_fft]"
                )
            den = _core.overlap_add_dual_dft_wola_den(
                analysis_window=analysis_window,
                n_fft=self.n_fft,
                frame_positions=frame_positions,
                signal_length=int(self._init_signal_length),
                eps=self.eps,
            )

            if method == "wola":
                num = _core.adstft_dft_adjoint(
                    stft=stft,
                    analysis_window=analysis_window.to(stft.dtype),
                    n_fft=self.n_fft,
                    frame_positions=frame_positions,
                    signal_length=int(self._init_signal_length),
                )
                return (num.real / den[None, :]).to(inv_dtype)

            # method == "cg": solve (A* A + λI)x = A* s with PCG.
            b = _core.adstft_dft_adjoint(
                stft=stft,
                analysis_window=analysis_window.to(stft.dtype),
                n_fft=self.n_fft,
                frame_positions=frame_positions,
                signal_length=int(self._init_signal_length),
            ).real

            def apply_a(x_time: torch.Tensor) -> torch.Tensor:
                frames_x, idx_floor_x, idx_frac_x = _core.unfold_floor_frac(
                    x=x_time,
                    frame_positions=frame_positions,
                    n_fft=self.n_fft,
                )
                return _core.adstft_dft_forward(
                    frames=frames_x,
                    analysis_window=analysis_window.to(x_time.dtype),
                    n_fft=self.n_fft,
                    idx_frac=idx_frac_x,
                    idx_floor=idx_floor_x,
                )

            def apply_at(y_stft: torch.Tensor) -> torch.Tensor:
                return _core.adstft_dft_adjoint(
                    stft=y_stft,
                    analysis_window=analysis_window.to(y_stft.dtype),
                    n_fft=self.n_fft,
                    frame_positions=frame_positions,
                    signal_length=int(self._init_signal_length),
                ).real

            def apply_mat(x_time: torch.Tensor) -> torch.Tensor:
                x_stft = apply_a(x_time)
                return apply_at(x_stft) + (float(cg_lambda) * x_time)

            def precond(r: torch.Tensor) -> torch.Tensor:
                return r / den[None, :]

            x0 = torch.zeros_like(b)
            x_hat = _core.cg_solve(
                apply_mat=apply_mat,
                b=b,
                x0=x0,
                max_iter=int(cg_max_iter),
                tol=float(cg_tol),
                precond=precond,
            )
            return x_hat.to(inv_dtype)

        # Undo the paper Eq. (25) phase convention.
        #
        # Forward (FFT backend) applies:
        #   spectr *= exp(-j 2π floor(t_n) m / n_fft)
        #
        # Therefore the inverse must apply the conjugate factor:
        #   spectr *= exp(+j 2π floor(t_n) m / n_fft)
        m = torch.arange(self.freq_bins, device=stft.device, dtype=inv_dtype)
        phase = torch.exp(
            (+2j * torch.pi / float(self.n_fft)) * (idx_floor[:, None] * m[None, :])
        )
        stft_unshift = stft.permute(0, 2, 1) * phase[None, :, :].to(stft.dtype)

        # Time-domain frames: [B, frames, n_fft]
        frames_td = torch.fft.irfft(stft_unshift, n=self.n_fft, dim=-1)

        return _core.overlap_add_dual(
            frames=frames_td,
            frame_positions=frame_positions.to(
                device=frames_td.device, dtype=frames_td.dtype
            ),
            analysis_window=analysis_window.to(
                device=frames_td.device, dtype=frames_td.dtype
            ),
            signal_length=int(self._init_signal_length),
            eps=self.eps,
        )

    def _inverse_dft_exact(self, stft: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DFT-backend inverse is temporarily disabled while the adjoint/Gram "
            "operator is being validated."
        )

        if self._init_signal_length is None:
            raise RuntimeError("DSTFT initialization metadata is missing")

        signal_length = int(self._init_signal_length)
        inv_dtype = stft.real.dtype
        frame_positions = self.frame_centers(device=stft.device, dtype=inv_dtype)
        idx_floor = frame_positions.floor().to(torch.int64)
        idx_frac = (frame_positions - idx_floor.to(frame_positions.dtype)).to(inv_dtype)
        theta = self._effective_win_length(device=stft.device, dtype=inv_dtype)

        analysis_window = self._window_fn(
            n_fft=self.n_fft,
            theta=theta,
            idx_frac=idx_frac,
            freq_bins=self.freq_bins,
            frames=frame_positions.numel(),
            device=stft.device,
            dtype=inv_dtype,
            normalization=self.normalization,
        )

        # Ensure full [F, T, L] for the adjoint.
        if analysis_window.ndim != 3:
            raise RuntimeError(
                "DFT inverse requires analysis_window with shape [freq_bins, frames, n_fft]"
            )

        def apply_gram(x_flat: torch.Tensor) -> torch.Tensor:
            x = x_flat.view(-1, signal_length)
            frames, _, _ = _core.unfold_floor_frac(
                x=x,
                frame_positions=frame_positions.to(device=x.device, dtype=x.dtype),
                n_fft=self.n_fft,
            )
            stft_x = _core.adstft_dft_forward(
                frames=frames,
                analysis_window=analysis_window.to(
                    device=frames.device, dtype=frames.dtype
                ),
                n_fft=self.n_fft,
                idx_frac=idx_frac.to(device=frames.device, dtype=frames.dtype),
                idx_floor=idx_floor,
            )
            y = _core.adstft_dft_adjoint(
                stft=stft_x,
                analysis_window=analysis_window.to(
                    device=frames.device, dtype=frames.dtype
                ),
                n_fft=self.n_fft,
                frame_positions=frame_positions.to(
                    device=frames.device, dtype=frames.dtype
                ),
                signal_length=signal_length,
            )
            return y.view_as(x_flat)

        b = _core.adstft_dft_adjoint(
            stft=stft,
            analysis_window=analysis_window.to(device=stft.device, dtype=inv_dtype),
            n_fft=self.n_fft,
            frame_positions=frame_positions.to(device=stft.device, dtype=inv_dtype),
            signal_length=signal_length,
        )

        x_hat = _core.cg_solve(
            apply_mat=apply_gram,
            b=b,
            x0=None,
            max_iter=200,
            tol=1e-10,
        )
        return x_hat

    def frame_centers(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the current frame center positions `t_n`.

        Convention: `t_n` are real-valued frame centers in the coordinate system
        of the input signal `x` (no explicit padding tensor).

        This is intended for debugging and visualization.

        Args:
            device: Optional device for the returned tensor.
            dtype: Optional dtype for the returned tensor.

        Returns:
            Tensor of shape `[frames]`.
        """
        if not self._initialized:
            raise RuntimeError("DSTFT must be initialized before reading frame centers")

        out_device = device if device is not None else self._init_device
        out_dtype = dtype if dtype is not None else self._init_dtype
        if out_device is None or out_dtype is None:
            raise RuntimeError("DSTFT initialization metadata is missing")

        if self._init_signal_length is None:
            raise RuntimeError("DSTFT initialization metadata is missing")

        if self.hop_mode == "time":
            hop = self._effective_hop_length(device=out_device, dtype=out_dtype)
            return hop.cumsum(dim=0)

        hop = self._effective_hop_length(device=out_device, dtype=out_dtype)
        return _core.compute_frame_positions_fixed_hop(
            signal_length=int(self._init_signal_length),
            n_fft=self.n_fft,
            hop_length=hop[0],
            device=out_device,
            dtype=out_dtype,
        )

    def _validate_parameter_shapes(self) -> None:
        if self._raw_win_length.ndim != 2:
            raise RuntimeError("win_length must be 2D")
        if self._num_frames is None:
            raise RuntimeError("DSTFT must be initialized to validate shapes")

        freq_dim, time_dim = self._raw_win_length.shape
        if freq_dim not in {1, self.freq_bins}:
            raise RuntimeError("win_length first dimension must be 1 or freq_bins")
        if time_dim not in {1, self._num_frames}:
            raise RuntimeError("win_length second dimension must be 1 or frames")

        if self._raw_hop_length.ndim != 1:
            raise RuntimeError("hop_length must be 1D")
        if self._raw_hop_length.shape[0] not in {1, self._num_frames}:
            raise RuntimeError("hop_length length must be 1 or frames")

    @property
    def hop_length(self) -> torch.Tensor:
        """Effective hop length (constrained) as a tensor."""
        if self._init_device is None or self._init_dtype is None:
            return self._effective_hop_length(
                device=self._raw_hop_length.device,
                dtype=self._raw_hop_length.dtype,
            )
        return self._effective_hop_length(
            device=self._init_device, dtype=self._init_dtype
        )

    @property
    def win_length(self) -> torch.Tensor:
        """Effective window length (constrained) as a tensor."""
        if self._init_device is None or self._init_dtype is None:
            return self._effective_win_length(
                device=self._raw_win_length.device,
                dtype=self._raw_win_length.dtype,
            )
        return self._effective_win_length(
            device=self._init_device, dtype=self._init_dtype
        )

    def _effective_win_length(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        raw = self._raw_win_length.to(device=device, dtype=dtype)
        win_min = torch.tensor(
            float(self._default_win_length_min), device=device, dtype=dtype
        )
        win_max = torch.tensor(float(self.n_fft), device=device, dtype=dtype)
        return win_min + (win_max - win_min) * torch.sigmoid(raw)

    def _effective_hop_length(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        raw = self._raw_hop_length.to(device=device, dtype=dtype)
        hop_min = torch.tensor(float(self.hop_length_min), device=device, dtype=dtype)
        hop_max = torch.tensor(float(self.hop_length_max), device=device, dtype=dtype)
        return hop_min + (hop_max - hop_min) * torch.sigmoid(raw)

    def plot_spec(self, spec: torch.Tensor, **kwargs):
        """Convenience wrapper around `dstft.visualization.plot_spec`.

        Args:
            spec: Magnitude spectrogram tensor of shape `[batch, freq, frames]`.
            **kwargs: Passed to `dstft.visualization.plot_spec`.

        Returns:
            `(fig, ax)` from matplotlib.
        """
        from .visualization import plot_spec

        return plot_spec(spec, **kwargs)

    def plot_win_lengths(self, **kwargs):
        """Convenience wrapper around `dstft.visualization.plot_win_lengths`.

        This uses the module's current window-length parameterization.

        Args:
            **kwargs: Passed to `dstft.visualization.plot_win_lengths`.

        Returns:
            `(fig, ax)` from matplotlib.
        """
        from .visualization import plot_win_lengths

        if self._init_device is None or self._init_dtype is None:
            raise RuntimeError("DSTFT must be initialized before plotting win_length")

        if "vmin" not in kwargs:
            kwargs["vmin"] = self.win_length_min
        if "vmax" not in kwargs:
            kwargs["vmax"] = self.win_length_max
        return plot_win_lengths(self.win_length, **kwargs)

    def _resolve_window(self, window: WindowSpec) -> WindowFn:
        if isinstance(window, str):
            if window == "hann":
                return windows.hann_window
            raise ValueError(f"Unknown window: {window!r}")
        if callable(window):
            return window
        raise TypeError("window must be a string identifier or a callable")

    # Normalization is applied in `dstft.windows` (window evaluation time).

    @torch.no_grad()
    def initialize(self, x: torch.Tensor) -> None:
        """Initialize input-dependent parameters.

        This method must be called once before the first `forward()` call.

        Args:
            x: Example input signal of shape `[batch, time]`.

        Raises:
            ValueError: If the input is too short for the configured `n_fft`.
            RuntimeError: If called again with a different signal length, device,
                or dtype.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"x must have shape [batch, time], got {x.shape}")

        signal_length = int(x.shape[-1])
        if signal_length < self.n_fft:
            raise ValueError(
                f"signal_length ({signal_length}) must be >= n_fft ({self.n_fft})"
            )

        if self._initialized:
            if self._init_signal_length != signal_length:
                raise RuntimeError(
                    "DSTFT is already initialized with signal_length="
                    f"{self._init_signal_length}, got {signal_length}. "
                    "Create a new DSTFT instance for a different signal length."
                )
            if self._init_device != x.device:
                raise RuntimeError(
                    "DSTFT is already initialized on device="
                    f"{self._init_device}, got {x.device}. "
                    "Move the module to the desired device before initializing."
                )
            if self._init_dtype != x.dtype:
                raise RuntimeError(
                    "DSTFT is already initialized with dtype="
                    f"{self._init_dtype}, got {x.dtype}. "
                    "Create a new DSTFT instance for a different dtype."
                )
            return

        # Cover policy: `t_n` are frame centers in the original signal
        # coordinates (no explicit padding tensor). We cover the full signal
        # length; frames outside the signal are implicitly zero-padded.
        # Use the current hop_length value (user-provided) to pick a stable frame
        # count. This must not depend on learnable unconstrained parameters.
        # Use the constructor hop length stored in the raw parameter to decide
        # a stable frame count.
        hop_value = float(
            (
                self.hop_length_min
                + (self.hop_length_max - self.hop_length_min)
                * torch.sigmoid(self._raw_hop_length.detach().cpu())[0]
            ).item()
        )
        hop_value = max(
            float(self.hop_length_min), min(hop_value, float(self.hop_length_max))
        )
        self._num_frames = int(1 + signal_length // hop_value)

        self._init_signal_length = signal_length
        self._init_device = x.device
        self._init_dtype = x.dtype

        # Initialize effective values from the constructor.
        freq_bins = self.freq_bins

        if self.window_mode == "time":
            self._raw_win_length = nn.Parameter(
                self._raw_win_length.detach()
                .to(device=x.device, dtype=x.dtype)
                .expand(1, self._num_frames)
                .clone(),
                requires_grad=True,
            )

        if self.window_mode in {"fixed", "constant"}:
            self._raw_win_length = nn.Parameter(
                self._raw_win_length.detach().to(device=x.device, dtype=x.dtype),
                requires_grad=(self.window_mode != "fixed"),
            )

        if self.window_mode == "frequency":
            self._raw_win_length = nn.Parameter(
                self._raw_win_length.detach()
                .to(device=x.device, dtype=x.dtype)
                .expand(freq_bins, 1)
                .clone(),
                requires_grad=True,
            )

        if self.window_mode == "time-frequency":
            self._raw_win_length = nn.Parameter(
                self._raw_win_length.detach()
                .to(device=x.device, dtype=x.dtype)
                .expand(freq_bins, self._num_frames)
                .clone(),
                requires_grad=True,
            )

        if self.hop_mode == "time":
            hop_raw = self._raw_hop_length.detach().to(device=x.device, dtype=x.dtype)
            self._raw_hop_length = nn.Parameter(
                hop_raw.expand(self._num_frames).clone(),
                requires_grad=True,
            )

        if self.hop_mode != "time":
            self._raw_hop_length = nn.Parameter(
                self._raw_hop_length.detach().to(device=x.device, dtype=x.dtype),
                requires_grad=(self.hop_mode != "fixed"),
            )

        self._validate_parameter_shapes()
        self._initialized = True
