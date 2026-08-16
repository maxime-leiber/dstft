"""Shared config for golden-output regression tests.

Not a test module itself (excluded from pytest collection by
`tool.pytest.ini_options.python_files = ["test_*.py"]`); imported by both
`tests/test_golden.py` and `tests/golden/_generate_references.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


GOLDEN_DIR = Path(__file__).parent

# Representative configs, one per backend/mode combination that has visibly
# different code paths in dstft.py/_core.py. Kept small (n_fft, signal
# length) so the whole suite stays fast, especially the DFT-backend configs
# (frequency/time-frequency), which are the most expensive.
CONFIGS: list[dict[str, Any]] = [
    {
        "name": "fft_constant_fixed_hop",
        "kwargs": {
            "n_fft": 64,
            "win_length": 48.0,
            "hop_length": 16.0,
            "window_mode": "constant",
            "hop_mode": "fixed",
        },
        "batch": 2,
        "length": 256,
    },
    {
        "name": "fft_time_fixed_hop",
        "kwargs": {
            "n_fft": 64,
            "win_length": 48.0,
            "hop_length": 16.0,
            "window_mode": "time",
            "hop_mode": "fixed",
        },
        "batch": 2,
        "length": 256,
    },
    {
        "name": "fft_constant_time_hop",
        "kwargs": {
            "n_fft": 64,
            "win_length": 48.0,
            "hop_length": 16.0,
            "window_mode": "constant",
            "hop_mode": "time",
        },
        "batch": 1,
        "length": 256,
    },
    {
        "name": "dft_frequency",
        "kwargs": {
            "n_fft": 32,
            "win_length": 24.0,
            "hop_length": 8.0,
            "window_mode": "frequency",
            "hop_mode": "fixed",
        },
        "batch": 1,
        "length": 128,
        # inverse() unconditionally raises ValueError for window_mode="frequency"
        # (analysis_window has a broadcastable frames dim of 1 there, but
        # overlap_add_dual_dft_wola_den requires an exact match) -- a real bug,
        # documented in TODO.md, not something to test the (nonexistent) output
        # of. Forward-only golden coverage until it's fixed.
        "test_inverse": False,
    },
    {
        "name": "dft_time_frequency",
        "kwargs": {
            "n_fft": 32,
            "win_length": 24.0,
            "hop_length": 8.0,
            "window_mode": "time-frequency",
            "hop_mode": "fixed",
        },
        "batch": 1,
        "length": 128,
    },
]


def make_signal(batch: int, length: int) -> torch.Tensor:
    """Build a deterministic multi-tone synthetic signal (no RNG involved).

    Avoiding `torch.randn`/`torch.manual_seed` keeps this independent of any
    RNG algorithm changes across torch versions -- the golden references
    should only change when dstft's own math changes, not torch's generator.
    """
    n = torch.arange(length, dtype=torch.float64)
    rows = []
    for b in range(batch):
        f1 = 0.03 + 0.01 * b
        f2 = 0.11 + 0.02 * b
        row = (
            torch.sin(2 * math.pi * f1 * n)
            + 0.5 * torch.sin(2 * math.pi * f2 * n + 0.3 * b)
            + 0.1 * torch.cos(2 * math.pi * 0.005 * n)
        )
        rows.append(row)
    return torch.stack(rows).to(torch.float32)


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"
