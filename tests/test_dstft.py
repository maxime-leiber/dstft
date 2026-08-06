from __future__ import annotations

import pytest
import torch

from dstft import DSTFT
from dstft.dstft import HopMode, WindowMode


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


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_fft": 0}, "n_fft must be positive"),
        ({"n_fft": -1}, "n_fft must be positive"),
        ({"hop_length": 0.0}, "hop_length must be positive"),
        ({"win_length": 0.0}, "win_length must be positive"),
        ({"magnitude_power": 0.0}, "magnitude_power must be positive"),
        ({"eps": 0.0}, "eps must be positive"),
    ],
)
def test_constructor_validates_parameters(kwargs: dict[str, float], match: str) -> None:
    kwargs.setdefault("n_fft", 64)
    with pytest.raises(ValueError, match=match):
        DSTFT(**kwargs)


def test_dstft_rejects_unknown_window_string() -> None:
    with pytest.raises(ValueError, match="Unknown window"):
        DSTFT(n_fft=64, window="not-hann")


def test_dstft_rejects_invalid_window_type() -> None:
    with pytest.raises(TypeError, match="window must be"):
        DSTFT(n_fft=64, window=123)  # type: ignore[arg-type]


def test_dstft_accepts_custom_window_callable() -> None:
    def rectangular_window(
        *,
        n_fft: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
        **_: object,
    ) -> torch.Tensor:
        return torch.ones(1, frames, n_fft, device=device, dtype=dtype)

    torch.manual_seed(0)
    x = torch.randn(1, 256)
    dstft = DSTFT(
        n_fft=64, hop_length=16.0, window=rectangular_window, window_mode="time"
    )
    dstft.initialize(x)
    spec, _ = dstft(x)
    assert torch.isfinite(spec).all()


# ---------------------------------------------------------------------------
# forward() across window_mode / hop_mode
# ---------------------------------------------------------------------------

WINDOW_MODES: list[WindowMode] = [
    "fixed",
    "constant",
    "time",
    "frequency",
    "time-frequency",
]


@pytest.mark.parametrize("window_mode", WINDOW_MODES)
def test_forward_across_window_modes(window_mode: WindowMode) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 256)
    dstft = DSTFT(n_fft=64, win_length=48.0, hop_length=16.0, window_mode=window_mode)
    dstft.initialize(x)

    spec, stft = dstft(x)

    assert spec.shape == stft.shape
    assert spec.shape[0] == 2
    assert torch.isfinite(spec).all()
    assert torch.isfinite(stft.abs()).all()


HOP_MODES: list[HopMode] = ["fixed", "constant", "time"]


@pytest.mark.parametrize("hop_mode", HOP_MODES)
def test_forward_across_hop_modes(hop_mode: HopMode) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 256)
    dstft = DSTFT(n_fft=64, win_length=48.0, hop_length=16.0, hop_mode=hop_mode)
    dstft.initialize(x)

    spec, _ = dstft(x)

    assert torch.isfinite(spec).all()


# ---------------------------------------------------------------------------
# forward() input validation
# ---------------------------------------------------------------------------


def test_forward_rejects_non_tensor_input() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        dstft([1.0, 2.0])  # type: ignore[arg-type]


def test_forward_rejects_wrong_ndim() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(ValueError, match=r"shape \[batch, time\]"):
        dstft(torch.randn(64))


def test_forward_before_initialize_raises() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(RuntimeError, match="must be initialized"):
        dstft(torch.randn(1, 256))


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


def test_initialize_rejects_signal_shorter_than_n_fft() -> None:
    dstft = DSTFT(n_fft=256)
    with pytest.raises(ValueError, match="must be >= n_fft"):
        dstft.initialize(torch.randn(1, 64))


def test_initialize_is_idempotent_for_same_signal_length() -> None:
    dstft = DSTFT(n_fft=64)
    dstft.initialize(torch.randn(1, 256))
    dstft.initialize(torch.randn(1, 256))  # same length, different values: no error


def test_initialize_rejects_different_signal_length() -> None:
    dstft = DSTFT(n_fft=64)
    dstft.initialize(torch.randn(1, 256))
    with pytest.raises(RuntimeError, match="already initialized with signal_length"):
        dstft.initialize(torch.randn(1, 128))


def test_initialize_rejects_different_dtype() -> None:
    dstft = DSTFT(n_fft=64)
    dstft.initialize(torch.randn(1, 256, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="already initialized with dtype"):
        dstft.initialize(torch.randn(1, 256, dtype=torch.float64))


# ---------------------------------------------------------------------------
# frame_centers() / win_length & hop_length properties
# ---------------------------------------------------------------------------


def test_frame_centers_before_initialize_raises() -> None:
    dstft = DSTFT(n_fft=64)
    with pytest.raises(
        RuntimeError, match="must be initialized before reading frame centers"
    ):
        dstft.frame_centers()


def test_frame_centers_with_time_hop_mode() -> None:
    dstft = DSTFT(n_fft=64, hop_mode="time")
    dstft.initialize(torch.randn(1, 256))
    centers = dstft.frame_centers()
    assert centers.ndim == 1
    assert torch.isfinite(centers).all()


def test_win_length_and_hop_length_properties_before_and_after_init() -> None:
    dstft = DSTFT(n_fft=64, win_length=48.0, hop_length=16.0)

    assert torch.isfinite(dstft.win_length).all()
    assert torch.isfinite(dstft.hop_length).all()

    dstft.initialize(torch.randn(1, 256))

    assert torch.isfinite(dstft.win_length).all()
    assert torch.isfinite(dstft.hop_length).all()
