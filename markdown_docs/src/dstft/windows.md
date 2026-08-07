# `dstft/windows.py`

Window functions for DSTFT. Currently a single function, `hann_window`, but
the module docstring states the intent explicitly: this file is meant to host
more window families later, with `hann_window`'s signature acting as the
template every future window function should match (a fixed keyword-only
signature is what lets `DSTFT` treat "which window" as a pluggable strategy —
see `WindowFn` / `WindowSpec` in `dstft.py`).

## `hann_window(*, n_fft, theta, idx_frac, freq_bins, frames, device, dtype, normalization) -> torch.Tensor`

Evaluates a Hann-family window whose **effective length is `theta`**, not
`n_fft`. This is the crux of the whole package: `theta` is an arbitrary
tensor (constant, or one value per time frame, or one per frequency bin, or
one per time-frequency cell) and can carry gradients, so an optimizer can
shrink or grow the window shape by adjusting `theta` directly.

**Shape contract.** `theta` must always be a 2D tensor
`[freq_bins_dim, frames_dim]` where each dimension is either `1`
(broadcast) or the true size — this uniform 2D-with-broadcast convention is
what lets the same function handle all four `window_mode` cases without
branching on the caller's side:

| `theta` shape | Resulting window shape | Corresponds to `window_mode` |
|---|---|---|
| `[1, 1]` (scalar) | `[1, 1, n_fft]` | `"fixed"` / `"constant"` |
| `[1, frames]` | `[1, frames, n_fft]` | `"time"` |
| `[freq_bins, 1]` | `[freq_bins, 1, n_fft]` | `"frequency"` |
| `[freq_bins, frames]` | `[freq_bins, frames, n_fft]` | `"time-frequency"` |

**How the window is actually evaluated.** The function builds a centered
integer support `k_rel = arange(n_fft) - n_fft/2`, subtracts the fractional
frame offset `idx_frac` (this is what lets a frame center land between
samples rather than only on integer sample indices — see `_core.py`'s
floor/frac split), and then evaluates a raised-cosine (Hann) shape on that
support. Two different formulas are used depending on `normalization`:

- `normalization in {"paper", "contract"}`: uses the exact contraction from
  the underlying paper, `ω(x, θ) = (L/θ) · ω_L((L/θ) · x)` where `L = n_fft`
  — i.e. the window is literally the fixed-length Hann window rescaled in
  both its argument and its amplitude by `n_fft / theta`. This is the
  mathematically "correct" contracted window and is also what keeps energy
  roughly constant as `theta` shrinks (the `* scale` amplitude term).
- Anything else (including `None`): a more direct formula, a Hann shape
  evaluated on a window of width `theta` directly, zero outside `[0, theta)`.
  Simpler, but does not preserve the same energy-normalization property as
  the paper's contraction.

After that, `normalization` is applied a second time, independently of the
shape formula above:

- `None` or `"paper"`/`"contract"` (already handled above): returned as-is.
- `"unit"`: the window is divided by its own sum along the last axis (with an
  epsilon floor to avoid division by zero), so each window sums to 1 —
  useful when you want the window's total energy/weight to be independent of
  `theta`.
- Anything else: raises `ValueError`.

**Why `idx_frac` and not just `theta`.** Note that `idx_frac` shifts *where*
the window is centered within its fixed `n_fft`-sample support, while `theta`
controls *how wide* the window is. These are independent degrees of freedom:
a frame can have a fractional center position (continuous frame placement)
independently of whatever its window length currently is.
