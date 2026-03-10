"""Visualization helpers for DSTFT.

This module assumes matplotlib is installed.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import torch


def plot_spec(
    spec: torch.Tensor,
    *,
    colorbar: bool = True,
    title: str | None = None,
    xlabel: str = "frames",
    ylabel: str = "frequency bins",
    cmap: str = "inferno",
    figsize: tuple[float, float] = (10.0, 4.0),
    ax: Any | None = None,
    show: bool = True,
    **imshow_kwargs: Any,
):
    """Plot a magnitude spectrogram.

    Args:
        spec: Spectrogram tensor of shape `[batch, freq, frames]`.
        colorbar: Whether to add a colorbar.
        title: Optional plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Matplotlib colormap.
        figsize: Figure size when creating a new figure.
        ax: Optional Matplotlib axis to plot on.
        show: Whether to call `plt.show()`.
        **imshow_kwargs: Forwarded to `ax.imshow(...)`.

    Returns:
        A tuple ``(fig, ax)``.

    Raises:
        ValueError: If ``spec`` does not have shape ``[batch, freq, frames]``.
    """
    if spec.ndim != 3:
        raise ValueError(
            f"spec must have shape [batch, freq, frames], got {spec.shape}"
        )

    data = spec[0].detach().cpu().to(torch.float32)
    data = (data + torch.finfo(data.dtype).eps).log()
    data_np = data.numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        data_np,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        **imshow_kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if colorbar:
        fig.colorbar(im, ax=ax)

    if show:
        plt.show()
    return fig, ax


def plot_win_lengths(
    win_length: float | torch.Tensor,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
    xlabel: str = "frames",
    ylabel: str = "frequency bins",
    cmap: str = "inferno",
    figsize: tuple[float, float] = (10.0, 4.0),
    ax: Any | None = None,
    show: bool = True,
):
    """Plot window length parameter(s).

    This plots a "distribution" image (like legacy versions):

    - scalar or 1D `[frames]` are shown as a 2D image with a singleton
      frequency axis.
    - 2D `[freq, frames]` is shown as-is.

    Returns:
        A tuple ``(fig, ax)``.

    Raises:
        ValueError: If ``win_length`` has an unsupported shape.
    """
    if isinstance(win_length, float | int):
        tensor = torch.tensor(float(win_length))
    else:
        tensor = win_length.detach()
    tensor = tensor.to(device="cpu", dtype=torch.float32)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if tensor.ndim == 0:
        image = tensor.view(1, 1)
    elif tensor.ndim == 1:
        image = tensor.view(1, -1)
    elif tensor.ndim == 2:
        image = tensor
    else:
        raise ValueError(
            "win_length must be a scalar, 1D [frames], or 2D [freq, frames] tensor"
        )

    im = ax.imshow(
        image.numpy(),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar:
        fig.colorbar(im, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
    return fig, ax
