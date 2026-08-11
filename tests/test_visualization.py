from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
import torch

from dstft import DSTFT
from dstft.visualization import plot_spec, plot_win_lengths


def test_plot_spec_returns_figure_and_axis() -> None:
    spec = torch.rand(1, 8, 10)
    fig, ax = plot_spec(spec, show=False)
    assert fig is not None
    assert ax is not None


def test_plot_spec_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match=r"shape \[batch, freq, frames\]"):
        plot_spec(torch.rand(8, 10), show=False)


def test_plot_spec_reuses_provided_axis() -> None:
    _, ax = plt.subplots()
    spec = torch.rand(1, 8, 10)
    _, returned_ax = plot_spec(spec, ax=ax, colorbar=False, title="t", show=False)
    assert returned_ax is ax


def test_plot_spec_calls_show(monkeypatch: pytest.MonkeyPatch) -> None:
    # matplotlib is forced to the non-interactive "Agg" backend in
    # conftest.py, so plt.show() is a no-op here rather than blocking.
    calls = []
    monkeypatch.setattr(
        "dstft.visualization.plt.show", lambda *a, **k: calls.append((a, k))
    )
    spec = torch.rand(1, 8, 10)
    fig, ax = plot_spec(spec, show=True)
    assert fig is not None
    assert ax is not None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "win_length",
    [
        256.0,
        torch.tensor(256.0),
        torch.full((5,), 256.0),
        torch.full((3, 5), 256.0),
    ],
)
def test_plot_win_lengths_shape_variants(
    win_length: float | torch.Tensor,
) -> None:
    fig, ax = plot_win_lengths(win_length, show=False)
    assert fig is not None
    assert ax is not None


def test_plot_win_lengths_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError, match="scalar, 1D"):
        plot_win_lengths(torch.rand(2, 3, 4), show=False)


def test_plot_win_lengths_reuses_provided_axis_and_title() -> None:
    _, ax = plt.subplots()
    fig, returned_ax = plot_win_lengths(256.0, ax=ax, title="t", show=False)
    assert returned_ax is ax
    assert fig is ax.figure
    assert returned_ax.get_title() == "t"


def test_plot_win_lengths_calls_show(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "dstft.visualization.plt.show", lambda *a, **k: calls.append((a, k))
    )
    fig, ax = plot_win_lengths(256.0, show=True)
    assert fig is not None
    assert ax is not None
    assert len(calls) == 1


def test_dstft_plot_wrappers() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 256)
    dstft = DSTFT(n_fft=64, win_length=48.0, hop_length=16.0, window_mode="constant")
    dstft.initialize(x)
    spec, _ = dstft(x)

    fig1, ax1 = dstft.plot_spec(spec, show=False)
    fig2, ax2 = dstft.plot_win_lengths(show=False)

    assert fig1 is not None
    assert ax1 is not None
    assert fig2 is not None
    assert ax2 is not None


def test_plot_win_lengths_before_initialize_raises() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(RuntimeError, match="must be initialized before plotting"):
        dstft.plot_win_lengths(show=False)
