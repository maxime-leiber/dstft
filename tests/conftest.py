from __future__ import annotations

import matplotlib
import pytest


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
