"""DSTFT core math."""

from __future__ import annotations

from math import pi

import torch

_FFT_FORWARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
_DFT_TWIDDLE_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_rfft_m(
    *, freq_bins: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (freq_bins, device, dtype)
    cached = _FFT_FORWARD_CACHE.get(key)
    if cached is None:
        cached = torch.arange(freq_bins, device=device, dtype=dtype)
        _FFT_FORWARD_CACHE[key] = cached
    return cached


def _get_dft_twiddle(
    *,
    n_fft: int,
    freq_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (n_fft, freq_bins, device, dtype)
    cached = _DFT_TWIDDLE_CACHE.get(key)
    if cached is None:
        k = torch.arange(n_fft, device=device, dtype=dtype)
        m = torch.arange(freq_bins, device=device, dtype=dtype)
        cached = torch.exp((-2j * pi / float(n_fft)) * (m[:, None] * k[None, :]))
        _DFT_TWIDDLE_CACHE[key] = cached
    return cached


def compute_frame_positions_fixed_hop(
    *,
    signal_length: int,
    n_fft: int,
    hop_length: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute frame center positions for a fixed hop (cover policy).

    Convention: `t_n` are true frame centers in the coordinate system of
    the input signal `x` (no explicit padding tensor). The window is evaluated
    on a centered discrete support of length `n_fft` around each `t_n`, and any
    out-of-range samples are implicitly treated as zeros.

    Shape policy ("cover"):
        `num_frames = 1 + floor(signal_length / hop_length)`

    This ensures we cover the full signal duration even though the last frames
    may extend beyond the signal boundaries (handled by implicit padding).

    This returns real-valued positions `t_n` as a Tensor (even if integer), so
    the API stays compatible with later learnable `t_n`.
    """
    hop = (
        hop_length.to(device=device, dtype=dtype)
        if isinstance(hop_length, torch.Tensor)
        else torch.tensor(float(hop_length), device=device, dtype=dtype)
    )
    length = torch.tensor(float(signal_length), device=device, dtype=dtype)
    num_frames = 1 + torch.floor(length / hop)

    # Shape policy: keep output shape stable by flooring to an integer count.
    # This is input-dependent and should not depend on learnable tensors.
    num_frames_i = int(num_frames.detach().cpu().item())
    return torch.arange(num_frames_i, device=device, dtype=dtype) * hop


def unfold_floor_frac(
    *,
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract frames using floor/frac decomposition.

    This is the core of the paper-aligned handling of non-integer frame
    positions: we separate integer indexing (via floor) from the fractional part
    (used downstream for window evaluation and phase correction).

    Returns:
        frames: [batch, frames, n_fft]
        idx_floor: [frames] (int64)
        idx_frac: [frames] (float)
    """
    idx_floor = frame_positions.floor().to(torch.int64)
    idx_frac = (frame_positions - idx_floor.to(frame_positions.dtype)).to(x.dtype)

    batch, signal_length = x.shape
    num_frames = frame_positions.numel()
    del batch
    del num_frames

    # Centered discrete support k ∈ {-n_fft//2, ..., n_fft//2-1}.
    half = int(n_fft) // 2
    k = torch.arange(n_fft, device=x.device, dtype=torch.int64) - half
    idx = idx_floor[:, None] + k[None, :]

    # Implicit zero-padding: sample outside [0, signal_length) are replaced by 0.
    valid = (idx >= 0) & (idx < signal_length)
    idx_safe = idx.clamp(0, signal_length - 1)

    gathered = x[:, idx_safe]  # [batch, frames, n_fft]
    frames = torch.where(valid[None, :, :], gathered, torch.zeros_like(gathered))
    return frames, idx_floor, idx_frac


def overlap_add_wola(
    *,
    frames: torch.Tensor,
    frame_positions: torch.Tensor,
    analysis_window: torch.Tensor,
    signal_length: int,
    eps: float,
) -> torch.Tensor:
    """Weighted overlap-add (WOLA) synthesis for DSTFT.

    This reconstructs a real-valued signal from time-domain frames using a
    differentiable weighted overlap-add (WOLA) scheme:

        x_hat = num / (den + eps)

    where `num` accumulates windowed frames and `den` accumulates the squared
    synthesis window (we use synthesis=analysis).

    Convention:
        Frames are assumed to be gathered/placed around `floor(t_n)` (local-frame
        convention), and any out-of-range samples are ignored (implicit
        zero-padding).

    Args:
        frames: Time-domain frames `[batch, frames, n_fft]`.
        frame_positions: Frame center positions `t_n` `[frames]`.
        analysis_window: Analysis window, broadcastable to `[1, frames, n_fft]`.
            Supported shapes: `[frames, n_fft]`, `[1, frames, n_fft]`, `[1, 1, n_fft]`.
        signal_length: Target output signal length.
        eps: Small constant for numerical stability.

    Returns:
        Reconstructed signal `x_hat` of shape `[batch, signal_length]`.
    """
    if frames.ndim != 3:
        raise ValueError("frames must have shape [batch, frames, n_fft]")
    if frame_positions.ndim != 1:
        raise ValueError("frame_positions must have shape [frames]")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    batch, num_frames, n_fft = frames.shape
    if num_frames != frame_positions.numel():
        raise ValueError("frames and frame_positions must agree on frames")

    if analysis_window.ndim == 2:
        win = analysis_window.unsqueeze(0)
    elif analysis_window.ndim == 3:
        win = analysis_window
    else:
        raise ValueError(
            "analysis_window must have shape [frames, n_fft] or [1, frames, n_fft] or [1, 1, n_fft]"
        )
    if win.shape[-1] != n_fft:
        raise ValueError("analysis_window last dimension must match n_fft")

    # Synthesis window equals analysis window (WOLA).
    win = win.to(device=frames.device, dtype=frames.dtype)
    win2 = win.pow(2)

    num = torch.zeros((batch, signal_length), device=frames.device, dtype=frames.dtype)
    den = torch.zeros((signal_length,), device=frames.device, dtype=frames.dtype)

    idx_floor = frame_positions.floor().to(torch.int64)
    half = int(n_fft) // 2
    k = torch.arange(n_fft, device=frames.device, dtype=torch.int64) - half
    idx = idx_floor[:, None] + k[None, :]  # [frames, n_fft]

    valid = (idx >= 0) & (idx < signal_length)
    idx_safe = idx.clamp(0, signal_length - 1)

    frames_win = frames * win
    frames_den = win2.squeeze(0) if win.shape[0] == 1 else win2

    for t in range(num_frames):
        idx_t = idx_safe[t]
        valid_t = valid[t]
        num[:, idx_t] += frames_win[:, t, :] * valid_t.to(frames.dtype)[None, :]

        # Denominator is batch-independent.
        den[idx_t] += frames_den[0 if frames_den.shape[0] == 1 else t, :] * valid_t.to(
            frames.dtype
        )

    return num / (den[None, :] + float(eps))


def overlap_add_dual(
    *,
    frames: torch.Tensor,
    frame_positions: torch.Tensor,
    analysis_window: torch.Tensor,
    signal_length: int,
    eps: float,
) -> torch.Tensor:
    """Overlap-add synthesis using a per-sample dual window.

    This implements an overlap-add synthesis with a pointwise normalization
    derived from the analysis window energy:

        x_hat[n] = (sum_t frame_t[k] * w_t[k]) / (sum_{t'} w_{t'}^2[k] + eps)

    where k indexes the local sample inside each frame, and the denominator is
    computed on the global time axis (with implicit zero-padding).

    This is required for stable reconstruction when the analysis window is
    variable (e.g. contracted inside the fixed `n_fft` support). Under the
    framing convention (implicit padding), exact reconstruction is expected on
    the covered region (typically up to `signal_length - n_fft//2`).
    """
    if frames.ndim != 3:
        raise ValueError("frames must have shape [batch, frames, n_fft]")
    if frame_positions.ndim != 1:
        raise ValueError("frame_positions must have shape [frames]")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    batch, num_frames, n_fft = frames.shape
    if num_frames != frame_positions.numel():
        raise ValueError("frames and frame_positions must agree on frames")

    if analysis_window.ndim == 2:
        win = analysis_window.unsqueeze(0)
    elif analysis_window.ndim == 3:
        win = analysis_window
    else:
        raise ValueError(
            "analysis_window must have shape [frames, n_fft] or [1, frames, n_fft] or [1, 1, n_fft]"
        )
    if win.shape[-1] != n_fft:
        raise ValueError("analysis_window last dimension must match n_fft")

    win = win.to(device=frames.device, dtype=frames.dtype)
    if win.shape[0] == 1:
        win = win.expand(1, num_frames, n_fft)
    elif win.shape[0] != num_frames:
        raise ValueError("analysis_window frames dimension must be 1 or match frames")

    idx_floor = frame_positions.floor().to(torch.int64)
    half = int(n_fft) // 2
    k = torch.arange(n_fft, device=frames.device, dtype=torch.int64) - half
    idx = idx_floor[:, None] + k[None, :]

    valid = (idx >= 0) & (idx < signal_length)
    idx_safe = idx.clamp(0, signal_length - 1)

    valid_f = valid.to(frames.dtype)
    idx_flat = idx_safe.reshape(-1)
    valid_flat = valid_f.reshape(-1)

    win_flat = win[0].reshape(-1)
    win2_flat = win_flat.pow(2) * valid_flat

    den = torch.zeros((signal_length,), device=frames.device, dtype=frames.dtype)
    den.index_add_(0, idx_flat, win2_flat)
    den = den.clamp_min(float(eps))

    weights_flat = (win_flat / den[idx_flat]) * valid_flat
    weighted_frames = frames.reshape(batch, -1) * weights_flat[None, :]

    out = torch.zeros((batch, signal_length), device=frames.device, dtype=frames.dtype)
    out.index_add_(1, idx_flat, weighted_frames)
    return out


def overlap_add_dual_dft_wola_den(
    *,
    analysis_window: torch.Tensor,
    n_fft: int,
    frame_positions: torch.Tensor,
    signal_length: int,
    eps: float,
) -> torch.Tensor:
    """Compute the DFT-backend WOLA denominator on the global time axis.

    For the DFT backend, the analysis window can vary across frequency bins.
    A fast approximate inverse can be formed as::

        x_hat ≈ A*(stft) / (den + eps)

    where the denominator corresponds to the diagonal of ``A* A``.

    Args:
        analysis_window: Window tensor `[freq_bins, frames, n_fft]`.
        n_fft: DFT size.
        frame_positions: Frame centers `[frames]`.
        signal_length: Output length.
        eps: Small positive number used for clamping.

    Returns:
        Denominator tensor `[signal_length]`.
    """
    if analysis_window.ndim != 3:
        raise ValueError("analysis_window must have shape [freq_bins, frames, n_fft]")
    if frame_positions.ndim != 1:
        raise ValueError("frame_positions must have shape [frames]")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")
    if analysis_window.shape[-1] != n_fft:
        raise ValueError("analysis_window last dimension must match n_fft")

    device = analysis_window.device
    dtype = analysis_window.dtype
    freq_bins, num_frames, _ = analysis_window.shape
    if frame_positions.numel() != num_frames:
        raise ValueError("frame_positions length must match analysis_window frames")

    # rFFT multiplicity weights: c_m = 1 for DC/Nyquist, 2 otherwise.
    c_m = torch.full((freq_bins,), 2.0, device=device, dtype=dtype)
    c_m[0] = 1.0
    if n_fft % 2 == 0 and freq_bins == (n_fft // 2 + 1):
        c_m[-1] = 1.0

    # Weighted sum over frequency of squared windows: [frames, n_fft]
    win2 = (analysis_window.abs().pow(2) * c_m[:, None, None]).sum(dim=0)

    idx_floor = frame_positions.floor().to(torch.int64)
    half = int(n_fft) // 2
    k = torch.arange(n_fft, device=device, dtype=torch.int64) - half
    idx = idx_floor[:, None] + k[None, :]

    valid = (idx >= 0) & (idx < signal_length)
    idx_safe = idx.clamp(0, signal_length - 1)

    idx_flat = idx_safe.reshape(-1)
    valid_flat = valid.to(dtype).reshape(-1)
    win2_flat = win2.reshape(-1) * valid_flat

    den = torch.zeros((signal_length,), device=device, dtype=dtype)
    den.index_add_(0, idx_flat, win2_flat)
    return den.clamp_min(float(eps))


def fdstft_fft_forward(
    *,
    frames: torch.Tensor,
    analysis_window: torch.Tensor,
    n_fft: int,
    idx_frac: torch.Tensor,
    idx_floor: torch.Tensor,
    fft_norm: None | str = None,
) -> torch.Tensor:
    """FFT-based DSTFT forward (FDSTFT).

    Args:
        frames: [batch, frames, n_fft]
        analysis_window: Broadcastable window on fixed support.
            Supported shapes:
            - [frames, n_fft]
            - [1, frames, n_fft]
            - [1, 1, n_fft]
        idx_frac: [frames] fractional part of the frame centers
        idx_floor: [frames] integer part of the frame centers

    Returns:
        stft: [batch, freq, frames]
    """
    if analysis_window.ndim == 2:
        win = analysis_window.unsqueeze(0)
    elif analysis_window.ndim == 3:
        win = analysis_window
    else:
        raise ValueError(
            "analysis_window must have shape [frames, n_fft] or [1, frames, n_fft] or [1, 1, n_fft]"
        )

    tapered = frames * win

    batch, num_frames, _ = tapered.shape
    # Avoid forcing contiguity: `.contiguous()` can create a large copy
    # and spike memory usage (especially on GPU).
    tapered_2d = tapered.reshape(batch * num_frames, n_fft)
    spectr = torch.fft.rfft(tapered_2d, n=n_fft, dim=-1, norm=fft_norm)  # [B*T, F]
    spectr = spectr.reshape(batch, num_frames, -1)

    # Paper Eq. (25) phase convention:
    #
    # After extracting a frame around floor(t_n) and shifting the window by
    # frac(t_n), Eq. (25) yields an additional frequency-domain phase factor
    # exp(-j 2π floor(t_n) m / L) when the complex exponential is written using
    # the *absolute* sample index. We apply that factor here.
    freq_bins = spectr.shape[-1]
    m = _get_rfft_m(freq_bins=freq_bins, device=frames.device, dtype=frames.dtype)
    phase = torch.exp((-2j * pi / float(n_fft)) * (idx_floor[:, None] * m[None, :]))
    spectr = spectr * phase[None, :, :]
    return spectr.permute(0, 2, 1)


def adstft_dft_forward(
    *,
    frames: torch.Tensor,
    analysis_window: torch.Tensor,
    n_fft: int,
    idx_frac: torch.Tensor,
    idx_floor: torch.Tensor,
) -> torch.Tensor:
    """DFT-based DSTFT forward (ADSTFT backend).

    This backend is used when the analysis window varies over frequency or
    time-frequency, making an FFT-based computation less convenient.

    Args:
        frames: [batch, frames, n_fft]
        analysis_window: [freq_bins, frames, n_fft]
        n_fft: DFT size
        idx_frac: [frames]
        idx_floor: [frames] integer part of the frame centers

    Returns:
        stft: [batch, freq_bins, frames]
    """
    # Paper Eq. (25): handle non-integer frame positions by (1) shifting the
    # window by frac(t_n) in the time domain and (2) applying the phase factor
    # exp(-j 2π floor(t_n) m / L) in the frequency domain.
    idx_frac = idx_frac.to(dtype=frames.dtype, device=frames.device)

    if analysis_window.ndim != 3:
        raise ValueError(
            "analysis_window must have shape [freq_bins, frames, n_fft] (with possible broadcastable freq_bins=1 or frames=1)"
        )
    if frames.ndim != 3:
        raise ValueError("frames must have shape [batch, frames, n_fft]")

    batch, num_frames, n_fft_frames = frames.shape
    freq_bins, window_frames, n_fft_win = analysis_window.shape
    if window_frames not in {1, num_frames}:
        raise ValueError(
            "analysis_window frames dimension must be 1 (broadcast) or match frames"
        )
    if n_fft_frames != n_fft or n_fft_win != n_fft:
        raise ValueError(
            "n_fft must match the last dimension of frames and analysis_window"
        )

    tapered = (
        frames[:, None, :, :] * analysis_window[None, :, :, :]
    )  # [B, F, T, L] (broadcast ok)

    # Convention note (paper Eq. 25): we follow the standard STFT convention
    # used by `torch.stft`, where the DFT is taken over the *local* frame index
    # k (after gathering samples around floor(t_n)). In that convention, no
    # extra exp(-j 2π floor(t_n) m / L) factor is applied.

    # DFT matrix for m=0..freq_bins-1 (rfft layout, paper-aligned sign).
    twiddle = _get_dft_twiddle(
        n_fft=n_fft,
        freq_bins=freq_bins,
        device=frames.device,
        dtype=frames.dtype,
    )

    tapered = tapered.to(twiddle.dtype).contiguous()
    # Sum over k.
    spectr = torch.einsum("bftk,fk->bft", tapered, twiddle)  # [B, F, T]

    m_phase = torch.arange(freq_bins, device=frames.device, dtype=frames.dtype)
    phase = torch.exp(
        (-2j * pi / float(n_fft)) * (idx_floor[:, None] * m_phase[None, :])
    )
    spectr = spectr * phase[None, :, :].transpose(-2, -1).to(spectr.dtype)

    del batch
    return spectr


def adstft_dft_adjoint(
    *,
    stft: torch.Tensor,
    analysis_window: torch.Tensor,
    n_fft: int,
    frame_positions: torch.Tensor,
    signal_length: int,
) -> torch.Tensor:
    """Adjoint operator of the DFT backend (exact, linear).

    This computes A*(stft) where A is `adstft_dft_forward` composed with the
    frame extraction operator.

    Convention:
        Uses the same local-frame convention as the forward pass: frames are
        gathered/placed around `floor(t_n)` on a centered support.

    Args:
        stft: Complex tensor `[batch, freq_bins, frames]`.
        analysis_window: Window tensor `[freq_bins, frames, n_fft]`.
        n_fft: DFT size.
        frame_positions: Frame centers `[frames]`.
        signal_length: Output length.

    Returns:
        Real tensor `[batch, signal_length]`.
    """
    if stft.ndim != 3:
        raise ValueError("stft must have shape [batch, freq_bins, frames]")
    if analysis_window.ndim != 3:
        raise ValueError("analysis_window must have shape [freq_bins, frames, n_fft]")
    if frame_positions.ndim != 1:
        raise ValueError("frame_positions must have shape [frames]")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    batch, freq_bins, num_frames = stft.shape
    win_f, win_t, win_l = analysis_window.shape
    if win_l != n_fft:
        raise ValueError("analysis_window last dimension must match n_fft")
    if win_f != freq_bins:
        raise ValueError("analysis_window freq_bins must match stft")
    if win_t != num_frames:
        raise ValueError("analysis_window frames must match stft")
    if frame_positions.numel() != num_frames:
        raise ValueError("frame_positions length must match stft frames")

    device = stft.device
    win = analysis_window.to(stft.dtype)

    # Undo the paper Eq. (25) phase convention (conjugate factor).
    idx_floor = frame_positions.floor().to(torch.int64)
    m_phase = torch.arange(freq_bins, device=device, dtype=torch.float32)
    phase = torch.exp(
        (+2j * pi / float(n_fft)) * (idx_floor[None, :] * m_phase[:, None])
    )
    stft = stft * phase[None, :, :].to(stft.dtype)

    # For rFFT bins, the adjoint of the truncated DFT includes multiplicity
    # weights: c_m=1 for DC (and Nyquist when present), 2 otherwise.
    c_m = torch.full((freq_bins,), 2.0, device=device, dtype=torch.float32)
    c_m[0] = 1.0
    if n_fft % 2 == 0 and freq_bins == (n_fft // 2 + 1):
        c_m[-1] = 1.0

    # Adjoint DFT (Hermitian transpose of the forward twiddle).
    twiddle_h = _get_dft_twiddle(
        n_fft=n_fft,
        freq_bins=freq_bins,
        device=device,
        dtype=torch.float32,
    ).conj()

    # A*(stft) per-frame/per-time-sample.
    frames_td = torch.einsum(
        "bft,fk,ftk->btk",
        stft,
        twiddle_h.to(stft.dtype),
        win,
    )

    # Scatter-add into the signal.
    half = int(n_fft) // 2
    k_idx = torch.arange(n_fft, device=device, dtype=torch.int64) - half
    idx = idx_floor[:, None] + k_idx[None, :]

    valid = (idx >= 0) & (idx < signal_length)
    idx_safe = idx.clamp(0, signal_length - 1)

    out = torch.zeros((batch, signal_length), device=device, dtype=frames_td.dtype)
    for t in range(num_frames):
        out[:, idx_safe[t]] += (
            frames_td[:, t, :] * valid[t].to(frames_td.dtype)[None, :]
        )

    return out


def cg_solve(
    *,
    apply_mat: callable,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
    precond: callable | None = None,
) -> torch.Tensor:
    """Conjugate gradient solver for symmetric positive definite operators.

    This is implemented for batched 2D tensors `[batch, dim]`.
    """
    if b.ndim != 2:
        raise ValueError("b must have shape [batch, dim]")
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_mat(x)
    z = r if precond is None else precond(r)
    p = z.clone()
    rs_old = (r * z).sum(dim=1, keepdim=True)
    b_norm = b.norm(dim=1, keepdim=True).clamp_min(torch.finfo(b.dtype).eps)

    for _ in range(int(max_iter)):
        Ap = apply_mat(p)
        denom = (p * Ap).sum(dim=1, keepdim=True)
        eps = torch.finfo(b.dtype).eps
        denom_safe = torch.where(denom.abs() < eps, denom.sign() * eps, denom)
        alpha = rs_old / denom_safe
        x = x + alpha * p
        r = r - alpha * Ap
        z = r if precond is None else precond(r)
        rs_new = (r * z).sum(dim=1, keepdim=True)
        if torch.all((rs_new.sqrt() / b_norm) < tol):
            break
        p = z + (rs_new / rs_old) * p
        rs_old = rs_new

    return x
