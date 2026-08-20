"""End-to-end integration test: a real gradient-descent optimization loop.

Unlike gradcheck (test_dstft.py), which only verifies the gradient is
locally correct at a single point, this runs several steps of a real
optimizer and checks the loss actually drops substantially over the run --
i.e. the gradient isn't just correct, it's usable for real training.
"""

from __future__ import annotations

import pytest
import torch

from dstft import DSTFT


@pytest.mark.integration
def test_optimizing_win_length_reduces_reconstruction_loss() -> None:
    """Gradient descent on a learnable win_length converges toward a target spectrogram."""
    torch.manual_seed(0)
    x = torch.randn(2, 512)

    target_win_length = 40.0
    target_model = DSTFT(
        n_fft=64,
        win_length=target_win_length,
        hop_length=16.0,
        window_mode="fixed",
        hop_mode="fixed",
    )
    target_model.initialize(x)
    with torch.no_grad():
        target_spec, _ = target_model(x)

    model = DSTFT(
        n_fft=64,
        win_length=16.0,
        hop_length=16.0,
        window_mode="constant",
        hop_mode="fixed",
    )
    model.initialize(x)
    initial_win_length = model.win_length.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        spec, _ = model(x)
        loss = torch.nn.functional.mse_loss(spec, target_spec)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Average over a few steps at each end rather than comparing single
    # first/last values: Adam overshoots and oscillates a bit right around
    # convergence, so a single noisy step shouldn't make this flaky.
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    assert final_loss < initial_loss * 0.1

    final_win_length = model.win_length.item()
    initial_distance = abs(initial_win_length - target_win_length)
    final_distance = abs(final_win_length - target_win_length)
    assert final_distance < initial_distance * 0.5

    # hop_mode="fixed" keeps hop_length non-learnable; the optimizer must not
    # have touched it (Adam skips params whose .grad stayed None).
    assert model.hop_length.item() == pytest.approx(16.0)
