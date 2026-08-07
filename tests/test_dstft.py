from __future__ import annotations

import pytest
import torch

import dstft
from dstft import DSTFT
from dstft.dstft import HopMode, WindowMode


def test_version_is_exposed() -> None:
    assert isinstance(dstft.__version__, str)
    assert dstft.__version__ != ""


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


# ---------------------------------------------------------------------------
# hop_length gradient flow with a scalar (fixed/constant) window
# ---------------------------------------------------------------------------
#
# With window_mode in {"fixed", "constant"}, the analysis window has a single
# shared shape ([1, 1, n_fft]) reused across every frame for performance. The
# window's *shape* only depends on hop_length through each frame's fractional
# offset (idx_frac); the phase correction applied afterward is phase-only and
# never affects a magnitude spectrogram. So if the shared window is built from
# only one frame's idx_frac, hop_length gets no (or an incomplete) gradient
# whenever a scalar window_mode is combined with a learnable hop_mode:
#
# - hop_mode="constant" (a single scalar hop, frame 0 always sits at position
#   0*hop=0 regardless of hop's value): gradient is exactly zero.
# - hop_mode="time" (one raw value per frame, frame_centers = hop.cumsum(0),
#   so frame_positions[0] == hop[0] directly): gradient is nonzero but only
#   for the first frame's own hop value, since only idx_frac[0] ever reaches
#   the shared window — every other frame's hop entry gets exactly zero
#   gradient despite being just as learnable.
#
# The fix should only pay the extra cost of a per-frame window when hop_mode
# actually needs it (hop_mode != "fixed"); hop_mode="fixed" must stay on the
# cheap shared path.


@pytest.mark.parametrize("window_mode", ["fixed", "constant"])
def test_hop_length_constant_gets_nonzero_gradient_with_scalar_window(
    window_mode: WindowMode,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 512)
    dstft = DSTFT(
        n_fft=128,
        win_length=64.0,
        hop_length=31.7,
        window_mode=window_mode,
        hop_mode="constant",
    )
    dstft.initialize(x)

    spec, _ = dstft(x)
    spec.sum().backward()

    assert dstft._raw_hop_length.grad is not None
    assert dstft._raw_hop_length.grad.norm().item() > 0.0


@pytest.mark.parametrize("window_mode", ["fixed", "constant"])
def test_hop_length_time_gets_gradient_from_every_frame_with_scalar_window(
    window_mode: WindowMode,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 512)
    dstft = DSTFT(
        n_fft=128,
        win_length=64.0,
        hop_length=31.7,
        window_mode=window_mode,
        hop_mode="time",
    )
    dstft.initialize(x)

    spec, _ = dstft(x)
    spec.sum().backward()

    grad = dstft._raw_hop_length.grad
    assert grad is not None
    # Not just frame 0: today only grad[0] is nonzero, everything else is
    # silently stuck at exactly zero.
    assert torch.all(grad != 0.0)


def test_fixed_hop_mode_keeps_the_shared_single_row_window() -> None:
    """hop_mode="fixed" must not pay for a per-frame window: no learnable hop
    means there's nothing for the extra per-frame cost to buy."""
    torch.manual_seed(0)
    x = torch.randn(1, 512)
    dstft = DSTFT(
        n_fft=128,
        win_length=64.0,
        hop_length=31.7,
        window_mode="fixed",
        hop_mode="fixed",
    )
    dstft.initialize(x)

    dstft(x)

    assert dstft.analysis_window.shape[1] == 1
