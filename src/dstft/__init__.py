"""DSTFT package.

The public API is the `DSTFT` class.
"""

from importlib.metadata import PackageNotFoundError, version

from .dstft import DSTFT
from .visualization import plot_spec, plot_win_lengths


try:
    __version__ = version("dstft")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["DSTFT", "__version__", "plot_spec", "plot_win_lengths"]
