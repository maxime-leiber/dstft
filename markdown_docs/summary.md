# dstft — project summary

Documentation of the codebase, module by module.

## What this package does

`dstft` implements a **Differentiable Short-Time Fourier Transform**: a
PyTorch `nn.Module` (`DSTFT`, in `dstft.py`) that computes a spectrogram the
same way `torch.stft` would, except every parameter that normally has to be
fixed up front — the analysis window's length, and the spacing between
frames — can instead be a learnable tensor. Because the whole transform is
implemented with differentiable PyTorch ops, gradients flow from a downstream
loss (e.g. "make this spectrogram sparse") back into the window length or hop
length, letting an optimizer discover a time-frequency tiling adapted to the
signal instead of a hand-picked one.

## Module map

| Module | Role |
|---|---|
| `dstft/__init__.py` | Public API surface: exports `DSTFT`, `plot_spec`, `plot_win_lengths`, `__version__`. |
| `dstft/dstft.py` | The `DSTFT` class itself — the only class users are expected to import directly. Holds configuration/state, validates inputs, and dispatches the actual math to `_core.py`. |
| `dstft/_core.py` | Pure-function math kernels: frame extraction, the two forward-transform backends (FFT-based and DFT-based), their adjoints, overlap-add reconstruction, and a small conjugate-gradient solver. No class state — everything is explicit tensors in, tensors out. |
| `dstft/windows.py` | Window functions. Currently one: a parameterized Hann window that can be evaluated at a continuous, learnable length `theta` and shifted by a fractional sample offset. |
| `dstft/visualization.py` | Two plotting helpers (`plot_spec`, `plot_win_lengths`) built on matplotlib, also exposed as convenience methods on `DSTFT`. |

## The two backends, and why there are two

`DSTFT.__init__` picks one of two transform implementations based on
`window_mode`, and this choice is **not** user-configurable directly — it's
implied by whether the window is allowed to vary across frequency:

- **FFT backend** (`_core.fdstft_fft_forward`) — used when `window_mode` is
  `"fixed"`, `"constant"`, or `"time"` (i.e. the window may change from frame
  to frame, but is the same for every frequency bin within a frame). This lets
  the implementation use one shared window per frame and a real FFT
  (`torch.fft.rfft`), which is fast.
- **DFT backend** (`_core.adstft_dft_forward`) — used when `window_mode` is
  `"frequency"` or `"time-frequency"`, i.e. the window itself depends on which
  frequency bin is being computed. An FFT can no longer be reused across bins
  in that case, so this backend falls back to an explicit matrix
  multiplication against a DFT twiddle-factor matrix (`torch.einsum`), which
  is more expensive but supports arbitrary per-bin windows.

Both backends implement the same non-integer-frame-position convention from
the underlying paper: a frame center `t_n` is split into an integer part
(`floor(t_n)`, used for indexing) and a fractional part (`frac(t_n)`, used to
continuously shift the window and to apply a phase-correction factor in the
frequency domain — see the "Eq. (25)" comments in `_core.py`). This is what
lets frame *positions* (via `hop_mode="time"`) be learnable too, not just
window length.

## Inversion

`DSTFT.inverse()` reconstructs a time-domain signal from a `stft` tensor.
Which method actually runs depends on the backend:

- On the FFT backend, inversion is **exact** on the covered region: the
  forward pass's per-frame windowing and phase convention are undone, then a
  weighted overlap-add (`_core.overlap_add_dual`) recombines frames using the
  analysis window as its own synthesis window, normalized by the local sum of
  squared window values. Because the numerator of that normalization equals
  `x[p] * (denominator)` for every covered sample `p`, the division exactly
  cancels and returns the original signal.
- On the DFT backend, exact inversion isn't available (the "frequency" and
  "time-frequency" window modes make the forward operator ill-conditioned to
  invert directly), so `inverse()` offers two approximations: `"wola"` (fast,
  applies the adjoint operator `_core.adstft_dft_adjoint` and normalizes by an
  approximate diagonal, i.e. a diagonal preconditioner) and `"cg"` (slower but
  more accurate — solves `(A*A + λI)x = A*s` with a preconditioned conjugate
  gradient solver, `_core.cg_solve`, using the same diagonal as a
  preconditioner).

See `notebooks/inverse.ipynb` for a worked example comparing these, and
`notebooks/window_modes.ipynb` for a tour of every `window_mode`/`hop_mode`
combination.
