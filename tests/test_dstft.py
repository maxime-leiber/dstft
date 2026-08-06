from __future__ import annotations

import torch

from dstft import DSTFT


def test_forward_returns_spec_and_stft() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 1024)

    dstft = DSTFT(
        n_fft=256,
        hop_length=64.0,
        win_length=256.0,
        window_mode="constant",
    )
    dstft.initialize(x)

    spec, stft = dstft(x)

    assert spec.shape[0] == 1
    assert stft.shape[0] == 1
    assert torch.isfinite(spec).all()
    assert torch.isfinite(stft.abs()).all()
