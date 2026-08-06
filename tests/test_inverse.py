from __future__ import annotations

import pytest
import torch

from dstft import DSTFT
from dstft.dstft import WindowMode


def _make_initialized(
    window_mode: WindowMode,
    *,
    n_fft: int = 64,
    win_length: float = 48.0,
    hop_length: float = 16.0,
    signal_length: int = 256,
) -> tuple[DSTFT, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(1, signal_length)
    dstft = DSTFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_mode=window_mode,
    )
    dstft.initialize(x)
    _, stft = dstft(x)
    return dstft, x, stft


@pytest.mark.parametrize("window_mode", ["fixed", "constant", "time"])
def test_inverse_fft_backend_reconstructs(window_mode: WindowMode) -> None:
    dstft, x, stft = _make_initialized(window_mode)
    x_hat = dstft.inverse(stft)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


@pytest.mark.parametrize("method", ["auto", "wola", "cg"])
def test_inverse_dft_backend_methods(method: str) -> None:
    dstft, x, stft = _make_initialized("time-frequency")
    x_hat = dstft.inverse(stft, method=method, cg_max_iter=5)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_inverse_rejects_non_tensor() -> None:
    dstft, _, _ = _make_initialized("constant")
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        dstft.inverse([1, 2, 3])  # type: ignore[arg-type]


def test_inverse_rejects_wrong_ndim() -> None:
    dstft, _, _ = _make_initialized("constant")
    with pytest.raises(ValueError, match=r"shape \[batch, freq, frames\]"):
        dstft.inverse(torch.randn(4, 4))


def test_inverse_before_initialize_raises() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(RuntimeError, match="must be initialized"):
        dstft.inverse(torch.randn(1, 33, 10, dtype=torch.cfloat))


def test_inverse_rejects_unknown_method() -> None:
    dstft, _, stft = _make_initialized("constant")
    with pytest.raises(ValueError, match="Unknown inverse method"):
        dstft.inverse(stft, method="nope")


def test_inverse_rejects_wrong_freq_dim() -> None:
    dstft, _, stft = _make_initialized("constant")
    bad_stft = stft[:, :-1, :]  # drop one frequency bin
    with pytest.raises(ValueError, match="freq dimension"):
        dstft.inverse(bad_stft)
