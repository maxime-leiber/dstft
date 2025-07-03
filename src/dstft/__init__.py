from .adstft import ADSTFT as ADSTFT
from .base import BaseSTFT
from .dstft import DSTFT as DSTFT

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import (  # type: ignore
        PackageNotFoundError,
        version,
    )

try:
    __version__ = version("dstft")
except PackageNotFoundError:
    __version__ = "0.2.0"  # fallback to pyproject.toml version

__all__ = ["ADSTFT", "DSTFT", "BaseSTFT"]
