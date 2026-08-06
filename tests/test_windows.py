from __future__ import annotations

import pytest
import torch

from dstft.windows import hann_window


def _call(
    theta: torch.Tensor,
    *,
    freq_bins: int = 1,
    frames: int = 4,
    normalization: str | None = None,
) -> torch.Tensor:
    idx_frac = torch.zeros(frames)
    return hann_window(
        n_fft=16,
        theta=theta,
        idx_frac=idx_frac,
        freq_bins=freq_bins,
        frames=frames,
        device=torch.device("cpu"),
        dtype=torch.float32,
        normalization=normalization,
    )


def test_hann_window_rejects_non_2d_theta() -> None:
    with pytest.raises(ValueError, match="theta must be 2D"):
        _call(torch.tensor(8.0))


def test_hann_window_rejects_bad_freq_dim() -> None:
    with pytest.raises(ValueError, match="first dimension"):
        _call(torch.full((3, 1), 8.0), freq_bins=5)


def test_hann_window_rejects_bad_time_dim() -> None:
    with pytest.raises(ValueError, match="second dimension"):
        _call(torch.full((1, 3), 8.0), frames=4)


def test_hann_window_rejects_frame_idx_frac_mismatch() -> None:
    with pytest.raises(ValueError, match="frames must match idx_frac"):
        hann_window(
            n_fft=16,
            theta=torch.full((1, 4), 8.0),
            idx_frac=torch.zeros(2),
            freq_bins=1,
            frames=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            normalization=None,
        )


@pytest.mark.parametrize("normalization", [None, "unit", "paper", "contract"])
def test_hann_window_normalization_modes(normalization: str | None) -> None:
    w = _call(torch.full((1, 1), 8.0), normalization=normalization)
    assert torch.isfinite(w).all()
    if normalization == "unit":
        assert torch.allclose(w.sum(dim=-1), torch.ones(1, 1), atol=1e-5)


def test_hann_window_paper_normalization_on_frequency_shape() -> None:
    # f_dim > 1, t_dim == 1: exercises the "frequency"/"time-frequency" branch
    # of the paper/contract scaling, distinct from the scalar/time branch.
    w = _call(torch.full((3, 1), 8.0), freq_bins=3, frames=4, normalization="paper")
    assert torch.isfinite(w).all()


def test_hann_window_rejects_unknown_normalization() -> None:
    with pytest.raises(ValueError, match="Unknown normalization"):
        _call(torch.full((1, 1), 8.0), normalization="bogus")


@pytest.mark.parametrize(
    ("f_dim", "t_dim"),
    [(1, 1), (1, 4), (3, 1), (3, 4)],
)
def test_hann_window_shape_variants(f_dim: int, t_dim: int) -> None:
    theta = torch.full((f_dim, t_dim), 8.0)
    w = _call(theta, freq_bins=3, frames=4)
    assert torch.isfinite(w).all()
    assert w.shape[-1] == 16
