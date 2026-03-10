"""DSTFT package.

The public API is the `DSTFT` class.
"""

from .dstft import DSTFT
from .visualization import plot_spec, plot_win_lengths


__all__ = ["DSTFT", "plot_spec", "plot_win_lengths"]
