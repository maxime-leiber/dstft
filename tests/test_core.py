"""Direct unit tests for `dstft._core`'s internal validation branches.

These functions are not part of the public API (see `dstft/__init__.py`),
but `DSTFT.forward()`/`DSTFT.inverse()` only ever call them with
already-consistent shapes, so their defensive `raise ValueError(...)` branches
are unreachable through the public API. Testing them directly here (rather
than trying to contort a public-API call into hitting them) is what actually
exercises the error-handling contract each function documents.
"""

from __future__ import annotations

import pytest
import torch

from dstft import _core


def _frames(batch: int = 1, frames: int = 3, n_fft: int = 8) -> torch.Tensor:
    return torch.randn(batch, frames, n_fft)


def _positions(frames: int = 3) -> torch.Tensor:
    return torch.arange(frames, dtype=torch.float32) * 4.0


# ---------------------------------------------------------------------------
# overlap_add_dual
# ---------------------------------------------------------------------------


def test_overlap_add_dual_rejects_wrong_frames_ndim() -> None:
    with pytest.raises(ValueError, match=r"frames must have shape"):
        _core.overlap_add_dual(
            frames=torch.randn(3, 8),
            frame_positions=_positions(),
            analysis_window=torch.ones(3, 8),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_wrong_frame_positions_ndim() -> None:
    with pytest.raises(ValueError, match=r"frame_positions must have shape"):
        _core.overlap_add_dual(
            frames=_frames(),
            frame_positions=_positions().unsqueeze(0),
            analysis_window=torch.ones(3, 8),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_non_positive_signal_length() -> None:
    with pytest.raises(ValueError, match="signal_length must be positive"):
        _core.overlap_add_dual(
            frames=_frames(),
            frame_positions=_positions(),
            analysis_window=torch.ones(3, 8),
            signal_length=0,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match="must agree on frames"):
        _core.overlap_add_dual(
            frames=_frames(frames=3),
            frame_positions=_positions(frames=4),
            analysis_window=torch.ones(3, 8),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_accepts_2d_window() -> None:
    # [frames, n_fft], no leading batch/broadcast dim.
    out = _core.overlap_add_dual(
        frames=_frames(),
        frame_positions=_positions(),
        analysis_window=torch.ones(3, 8),
        signal_length=16,
        eps=1e-8,
    )
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()


def test_overlap_add_dual_rejects_bad_window_ndim() -> None:
    with pytest.raises(ValueError, match="analysis_window must have shape"):
        _core.overlap_add_dual(
            frames=_frames(),
            frame_positions=_positions(),
            analysis_window=torch.ones(8),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_window_n_fft_mismatch() -> None:
    with pytest.raises(ValueError, match="last dimension must match n_fft"):
        _core.overlap_add_dual(
            frames=_frames(n_fft=8),
            frame_positions=_positions(),
            analysis_window=torch.ones(3, 4),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_invalid_leading_dim() -> None:
    with pytest.raises(ValueError, match="leading dimension must be 1"):
        _core.overlap_add_dual(
            frames=_frames(frames=3),
            frame_positions=_positions(frames=3),
            analysis_window=torch.ones(2, 3, 8),  # leading dim 2 != 1
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_rejects_window_frame_count_mismatch() -> None:
    # Regression test: shape[0]==1 (valid leading dim) but shape[1] (the
    # actual frames axis) is neither 1 nor num_frames used to fall through
    # to `.expand()` and raise RuntimeError instead of this ValueError
    # (see TODO.md [+coverage-95] / PR #26's description).
    with pytest.raises(ValueError, match="frames dimension must be 1 or match frames"):
        _core.overlap_add_dual(
            frames=_frames(frames=3),
            frame_positions=_positions(frames=3),
            analysis_window=torch.ones(1, 2, 8),  # frames dim 2 != 1 and != 3
            signal_length=16,
            eps=1e-8,
        )


# ---------------------------------------------------------------------------
# overlap_add_dual_dft_wola_den
# ---------------------------------------------------------------------------


def test_overlap_add_dual_dft_wola_den_rejects_bad_window_ndim() -> None:
    with pytest.raises(ValueError, match=r"analysis_window must have shape"):
        _core.overlap_add_dual_dft_wola_den(
            analysis_window=torch.ones(3, 8),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_dft_wola_den_rejects_bad_frame_positions_ndim() -> None:
    with pytest.raises(ValueError, match=r"frame_positions must have shape"):
        _core.overlap_add_dual_dft_wola_den(
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions().unsqueeze(0),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_dft_wola_den_rejects_non_positive_signal_length() -> None:
    with pytest.raises(ValueError, match="signal_length must be positive"):
        _core.overlap_add_dual_dft_wola_den(
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=0,
            eps=1e-8,
        )


def test_overlap_add_dual_dft_wola_den_rejects_n_fft_mismatch() -> None:
    with pytest.raises(ValueError, match="last dimension must match n_fft"):
        _core.overlap_add_dual_dft_wola_den(
            analysis_window=torch.ones(5, 3, 4),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
            eps=1e-8,
        )


def test_overlap_add_dual_dft_wola_den_rejects_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match="frame_positions length must match"):
        _core.overlap_add_dual_dft_wola_den(
            analysis_window=torch.ones(5, 4, 8),
            n_fft=8,
            frame_positions=_positions(frames=3),
            signal_length=16,
            eps=1e-8,
        )


# ---------------------------------------------------------------------------
# fdstft_fft_forward
# ---------------------------------------------------------------------------


def test_fdstft_fft_forward_accepts_2d_window() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    spectr = _core.fdstft_fft_forward(
        frames=_frames(),
        analysis_window=torch.ones(3, 8),
        n_fft=8,
        idx_frac=torch.zeros(3),
        idx_floor=idx_floor,
    )
    assert spectr.shape[0] == 1
    assert spectr.shape[-1] == 3


def test_fdstft_fft_forward_rejects_bad_window_ndim() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    with pytest.raises(ValueError, match="analysis_window must have shape"):
        _core.fdstft_fft_forward(
            frames=_frames(),
            analysis_window=torch.ones(8),
            n_fft=8,
            idx_frac=torch.zeros(3),
            idx_floor=idx_floor,
        )


# ---------------------------------------------------------------------------
# adstft_dft_forward
# ---------------------------------------------------------------------------


def test_adstft_dft_forward_rejects_bad_window_ndim() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    with pytest.raises(ValueError, match="analysis_window must have shape"):
        _core.adstft_dft_forward(
            frames=_frames(),
            analysis_window=torch.ones(3, 8),  # 2D, must be 3D
            n_fft=8,
            idx_frac=torch.zeros(3),
            idx_floor=idx_floor,
        )


def test_adstft_dft_forward_rejects_bad_frames_ndim() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    with pytest.raises(ValueError, match="frames must have shape"):
        _core.adstft_dft_forward(
            frames=torch.randn(1, 8),
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            idx_frac=torch.zeros(3),
            idx_floor=idx_floor,
        )


def test_adstft_dft_forward_rejects_window_frame_count_mismatch() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    with pytest.raises(ValueError, match="frames dimension must be 1"):
        _core.adstft_dft_forward(
            frames=_frames(frames=3),
            analysis_window=torch.ones(5, 4, 8),  # 4 frames != 3 and != 1
            n_fft=8,
            idx_frac=torch.zeros(3),
            idx_floor=idx_floor,
        )


def test_adstft_dft_forward_rejects_n_fft_mismatch() -> None:
    idx_floor = _positions().floor().to(torch.int64)
    with pytest.raises(ValueError, match="n_fft must match"):
        _core.adstft_dft_forward(
            frames=_frames(n_fft=8),
            analysis_window=torch.ones(5, 3, 4),  # window n_fft=4 != 8
            n_fft=8,
            idx_frac=torch.zeros(3),
            idx_floor=idx_floor,
        )


# ---------------------------------------------------------------------------
# adstft_dft_adjoint
# ---------------------------------------------------------------------------


def _stft(freq_bins: int = 5, frames: int = 3) -> torch.Tensor:
    return torch.randn(1, freq_bins, frames, dtype=torch.cfloat)


def test_adstft_dft_adjoint_rejects_bad_stft_ndim() -> None:
    with pytest.raises(ValueError, match=r"stft must have shape"):
        _core.adstft_dft_adjoint(
            stft=torch.randn(5, 3, dtype=torch.cfloat),
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_bad_window_ndim() -> None:
    with pytest.raises(ValueError, match="analysis_window must have shape"):
        _core.adstft_dft_adjoint(
            stft=_stft(),
            analysis_window=torch.ones(5, 8),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_bad_frame_positions_ndim() -> None:
    with pytest.raises(ValueError, match=r"frame_positions must have shape"):
        _core.adstft_dft_adjoint(
            stft=_stft(),
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions().unsqueeze(0),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_non_positive_signal_length() -> None:
    with pytest.raises(ValueError, match="signal_length must be positive"):
        _core.adstft_dft_adjoint(
            stft=_stft(),
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions(),
            signal_length=0,
        )


def test_adstft_dft_adjoint_rejects_n_fft_mismatch() -> None:
    with pytest.raises(ValueError, match="last dimension must match n_fft"):
        _core.adstft_dft_adjoint(
            stft=_stft(),
            analysis_window=torch.ones(5, 3, 4),  # window n_fft=4 != 8
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_freq_bins_mismatch() -> None:
    with pytest.raises(ValueError, match="freq_bins must match stft"):
        _core.adstft_dft_adjoint(
            stft=_stft(freq_bins=5),
            analysis_window=torch.ones(4, 3, 8),  # 4 != 5
            n_fft=8,
            frame_positions=_positions(),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_window_frames_mismatch() -> None:
    with pytest.raises(ValueError, match="frames must match stft"):
        _core.adstft_dft_adjoint(
            stft=_stft(frames=3),
            analysis_window=torch.ones(5, 4, 8),  # 4 != 3
            n_fft=8,
            frame_positions=_positions(frames=3),
            signal_length=16,
        )


def test_adstft_dft_adjoint_rejects_frame_positions_length_mismatch() -> None:
    with pytest.raises(ValueError, match="frame_positions length must match"):
        _core.adstft_dft_adjoint(
            stft=_stft(frames=3),
            analysis_window=torch.ones(5, 3, 8),
            n_fft=8,
            frame_positions=_positions(frames=4),  # 4 != 3
            signal_length=16,
        )


# ---------------------------------------------------------------------------
# cg_solve
# ---------------------------------------------------------------------------


def test_cg_solve_rejects_bad_b_ndim() -> None:
    with pytest.raises(ValueError, match=r"b must have shape \[batch, dim\]"):
        _core.cg_solve(apply_mat=lambda x: x, b=torch.randn(4))


def test_cg_solve_converges_early_on_identity_system() -> None:
    # apply_mat = identity => exact solution in one CG iteration, well before
    # max_iter is reached. This exercises the early-`break` convergence path.
    b = torch.randn(2, 6)
    x = _core.cg_solve(apply_mat=lambda v: v, b=b, max_iter=200, tol=1e-6)
    assert torch.allclose(x, b, atol=1e-4)
