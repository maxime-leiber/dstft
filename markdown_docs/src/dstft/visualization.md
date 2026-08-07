# `dstft/visualization.py`

Two matplotlib-based plotting helpers. Both are plain functions (no
dependency on `DSTFT` itself — they take tensors, not a module instance),
and both are also re-exported as convenience methods on `DSTFT`
(`DSTFT.plot_spec`, `DSTFT.plot_win_lengths` in `dstft.py`), which just
forward to these with a couple of `DSTFT`-derived defaults filled in
(`vmin`/`vmax` from `win_length_min`/`win_length_max`).

## `plot_spec(spec, *, colorbar=True, title=None, xlabel="frames", ylabel="frequency bins", cmap="inferno", figsize=(10.0, 4.0), ax=None, show=True, **imshow_kwargs) -> (fig, ax)`

Plots a magnitude spectrogram as a heatmap. Expects `spec` with shape
`[batch, freq, frames]` and only plots the **first item of the batch**
(`spec[0]`) — this is a single-spectrogram debugging/inspection plot, not a
batch-aware grid.

Before plotting, the data is detached, moved to CPU, cast to `float32`, and
put on a **log scale** (`(data + eps).log()`, using the dtype's own epsilon
to avoid `log(0)`). This log-magnitude view is standard for spectrograms —
raw linear magnitude is dominated by a few loud bins and hides everything
else.

Accepts an existing `ax` to plot into (for building multi-panel figures —
see how `notebooks/window_modes.ipynb` uses this to lay out several modes
side by side in one figure) or creates a new one via `plt.subplots(figsize)`
if none is given. Any extra keyword arguments are forwarded straight to
`ax.imshow(...)`, so callers can override `vmin`/`vmax`/`interpolation`/etc.
without this function needing to know about them explicitly.

## `plot_win_lengths(win_length, *, vmin=None, vmax=None, colorbar=True, title=None, xlabel="frames", ylabel="frequency bins", cmap="inferno", figsize=(10.0, 4.0), ax=None, show=True) -> (fig, ax)`

Plots a window-length (or hop-length — the function is generic over
whichever tensor you pass it) parameter as a 2D heatmap, regardless of its
actual rank. `win_length` can be:

- a plain Python `float`/`int` (a scalar `theta`, e.g. `window_mode="fixed"`
  or `"constant"`) — reshaped to a `[1, 1]` image,
- a 1D tensor `[frames]` (`window_mode="time"`, or a hop-length tensor) —
  reshaped to `[1, frames]`,
- a 2D tensor `[freq_bins, frames]` (`window_mode="frequency"` or
  `"time-frequency"`) — plotted as-is.

Any other rank raises `ValueError`. This "always show it as a 2D image, even
a 1x1 or 1xN one" approach is what lets one function visualize the parameter
regardless of which `window_mode` produced it, without the caller having to
branch.

Note that unlike `plot_spec`, this function does **not** apply a log
transform — window/hop lengths are plotted on a linear scale, which is why
`vmin`/`vmax` are exposed as explicit parameters (so callers can pin the
color scale to the module's configured `win_length_min`/`win_length_max`
bounds, which is exactly what `DSTFT.plot_win_lengths` does by default).
