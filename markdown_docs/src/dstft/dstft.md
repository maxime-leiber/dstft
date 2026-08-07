# `dstft/dstft.py`

The single user-facing module. Contains the `DSTFT` class and the small
supporting type aliases/protocol that describe its configuration surface.
Per the module docstring, this file is deliberately an "API/state holder" —
it validates inputs, manages learnable parameters, and picks which backend
to call, but the actual transform math lives in `_core.py`.

## Type aliases and `WindowFn`

- `Normalization = None | Literal["unit", "paper", "contract"]` — passed
  through to the window function; see `windows.md`.
- `WindowMode = Literal["fixed", "constant", "time", "frequency", "time-frequency"]`
  — controls both which of `theta`'s two axes (frequency, time) are allowed
  to vary, and, as a side effect, which backend gets selected (see below).
- `HopMode = Literal["fixed", "constant", "time"]` — the hop-length analog of
  `WindowMode`, but with only a time axis (there's no "hop length that varies
  by frequency bin" — hop length determines frame *positions*, which are
  shared across all frequency bins of a frame).
- `WindowFn` (a `Protocol`) — the exact call signature any window function
  must implement to be pluggable into `DSTFT(window=...)`. `hann_window` in
  `windows.py` is the only implementation today, but this protocol is what
  would let a caller supply their own window shape without modifying this
  file.

## `class DSTFT(nn.Module)`

### `__init__(self, *, n_fft, win_length=None, hop_length=None, window_mode="time", hop_mode="fixed", window="hann", normalization=None, hop_length_min=1.0, hop_length_max=None, win_length_min=None, win_length_max=None, magnitude_power=1.0, eps=1e-12)`

Validates all the numeric arguments (positivity checks on `n_fft`,
`hop_length`, `win_length`, `magnitude_power`, `eps`), then sets up two kinds
of state:

1. **Configuration** — plain Python attributes (`n_fft`, `freq_bins =
   n_fft // 2 + 1`, `window_mode`, `hop_mode`, the min/max bounds, etc.), all
   fixed for the module's lifetime.
2. **Learnable parameters, stored in an unconstrained form.** `win_length`
   and `hop_length` are not stored directly as `nn.Parameter`s in their
   user-facing units. Instead, `_raw_win_length` / `_raw_hop_length` hold an
   unconstrained real number that gets mapped through a `sigmoid`, then
   affinely rescaled into `[min, max]` — see `_effective_win_length` /
   `_effective_hop_length` below. This is a standard reparameterization trick
   for **bounded** learnable parameters: an unconstrained value can be
   optimized with plain gradient descent without ever needing a projection
   step to keep it in-bounds, because the sigmoid does that automatically.
   The constructor solves for the initial raw value that makes
   `sigmoid(raw)` produce the user's requested initial `win_length`/
   `hop_length` exactly (the `torch.log(p) - torch.log1p(-p)` lines are the
   logit function, i.e. `sigmoid`'s inverse).

   Whether these parameters actually require gradients is decided by
   `window_mode`/`hop_mode`: `"fixed"` never requires grad; anything else
   does. Note this means `"constant"` *is* learnable (a single scalar that
   the whole module shares) — the difference from `"fixed"` is trainability,
   not shape.

   Frame-shaped parameters (one value per frame, or per frequency bin, or
   both) are **not** allocated here — their size depends on the input
   signal's length, which isn't known until `initialize(x)` is called. The
   constructor only allocates a placeholder scalar; `initialize()` expands it
   to the right shape.

3. **Backend dispatch, resolved once.** `self._transform_fn` is bound to
   either `_core.adstft_dft_forward` (if `window_mode` is `"frequency"` or
   `"time-frequency"`) or `_core.fdstft_fft_forward` (otherwise) — decided
   once in `__init__`, not re-checked on every `forward()` call, so there's
   no per-call branching cost. Same idea for `self._window_fn`, resolved via
   `_resolve_window`.

### `forward(self, x) -> (spec, stft)`

The main transform. Requires `initialize(x)` to have been called first (this
is enforced with a `RuntimeError`, not silently auto-initializing — the
docstring is explicit that this is deliberate, since auto-initializing on
first `forward()` could silently lock in a signal length the caller didn't
intend to commit to). Steps:

1. Compute frame center positions via `self.frame_centers(...)`.
2. Extract raw frames and the floor/frac split via `_core.unfold_floor_frac`.
3. Evaluate the analysis window at the module's current *effective* window
   length (`self._effective_win_length(...)`, the sigmoid-mapped `theta`) via
   `self._window_fn`.
4. A shape sanity-check on the returned window, branching on `window_mode`
   (2D/3D for the FFT-eligible modes, always 3D for the DFT modes) — this is
   defensive validation against a custom `window=` callable returning the
   wrong rank, not something that fires with the built-in `hann_window`.
5. Dispatch to `self._transform_fn` (one of the two `_core` backends).
6. Compute the magnitude spectrogram: `spec = stft.abs().pow(magnitude_power)
   + eps` — note the `eps` is added *after* the power, as a floor, primarily
   so a downstream `.log()` (as used in `visualization.plot_spec`) never sees
   an exact zero.

Returns both `spec` (real, `[batch, freq, frames]`) and `stft` (complex,
same shape) — the complex `stft` is what you need to pass to `inverse()`
later; `spec` alone has thrown away the phase.

### `inverse(self, stft, *, method="auto", cg_max_iter=20, cg_tol=1e-8, cg_lambda=1e-6) -> Tensor[batch, time]`

Reconstructs a time-domain signal from a complex `stft` tensor. The
docstring on this method is unusually detailed for the codebase and is worth
reading directly for the exact reconstruction identity — see also
`_core.md` for how `overlap_add_dual` achieves exact recovery. In brief:

- `method="auto"` resolves to `"wola"` on the FFT backend (exact) or `"cg"`
  on the DFT backend (approximate but more accurate than `"wola"` there).
- `method="wola"` on the FFT backend: undoes the forward pass's Eq. (25)
  phase factor, runs `torch.fft.irfft` to get back per-frame time-domain
  frames, then calls `_core.overlap_add_dual` to recombine them — this is the
  path that reconstructs the original signal exactly (on the region actually
  covered by at least one frame).
- On the DFT backend (`window_mode` in `{"frequency", "time-frequency"}`),
  neither `"wola"` nor `"cg"` are exact — the operator isn't easily
  invertible when every frequency bin has its own window. `"wola"` applies
  the adjoint (`_core.adstft_dft_adjoint`) and normalizes by an approximate
  diagonal (fast, one adjoint call). `"cg"` instead solves
  `(A*A + λI) x = A*s` iteratively with `_core.cg_solve`, using the same
  diagonal as a preconditioner — more accurate, more expensive
  (`cg_max_iter` iterations of forward+adjoint each).

Note there is also an `_inverse_dft_exact` method further down the file that
currently just raises `NotImplementedError` unconditionally (with dead code
below the `raise` that was presumably a work-in-progress exact DFT-backend
inverse via CG against the full Gram operator, left in place but disabled —
worth knowing about if `"cg"`'s accuracy on the DFT backend is ever
insufficient and someone wants to pick this back up).

### `frame_centers(self, *, device=None, dtype=None) -> Tensor[frames]`

Returns the current frame center positions `t_n`. Must be called after
`initialize()`. If `hop_mode == "time"`, positions come from a **cumulative
sum** of the (learnable, per-frame) effective hop lengths — i.e. each frame's
position is literally "wherever the previous frame ended up, plus this
frame's own hop" — which is what makes hop *lengths* learnable translate into
learnable frame *positions* without ever storing positions directly as
parameters. Otherwise (fixed/constant hop), positions come from
`_core.compute_frame_positions_fixed_hop` using the single effective hop
value.

### `hop_length` / `win_length` (properties)

Public read accessors for the *effective* (constrained, sigmoid-mapped)
values — never the raw unconstrained parameters. This is the only way
user code is meant to read the current window/hop length; `_raw_win_length`
and `_raw_hop_length` are private.

### `_effective_win_length` / `_effective_hop_length`

The sigmoid reparameterization itself:
`min + (max - min) * sigmoid(raw)`. Small, private, and called from several
places (`forward`, `inverse`, the public properties above) — this is the
single source of truth for "what does the raw parameter currently mean in
real units."

### `plot_spec` / `plot_win_lengths`

Thin instance-method wrappers around `visualization.plot_spec` /
`visualization.plot_win_lengths` (imported lazily, inside the method, to
avoid importing matplotlib at module import time for users who never plot).
`plot_win_lengths` additionally fills in `vmin`/`vmax` from the module's own
`win_length_min`/`win_length_max` if the caller didn't specify them, and
requires the module to already be initialized (it needs `_init_device`/
`_init_dtype` to read `self.win_length`).

### `_resolve_window(self, window) -> WindowFn`

Converts the constructor's `window=` argument (a string identifier or a
callable) into an actual callable. Today the only supported string is
`"hann"`, mapped to `windows.hann_window`; any other string raises
`ValueError`, and anything callable is accepted as-is (assumed to satisfy
the `WindowFn` protocol — not checked at runtime beyond `callable(window)`).

### `initialize(self, x)`

Must be called once, before the first `forward()`/`inverse()`, with an
example input `x: [batch, time]`. This is where every input-length-dependent
piece of state actually gets allocated:

- Validates `x` is 2D and at least `n_fft` samples long.
- If already initialized, this call becomes idempotent **only if** the new
  `x` has the same signal length, device, and dtype as before — any mismatch
  raises `RuntimeError` rather than silently re-initializing (the module is
  explicitly meant to be reused across forward passes on same-shaped inputs
  from the same batch/dataset, not to silently support changing input shapes
  mid-training).
- Computes `self._num_frames` from a **detached** hop-length value (see the
  comment in the source: frame *count* must not depend on a
  gradient-tracked tensor, since shapes participate in the computation
  graph's structure, not its values).
- Expands `_raw_win_length` and `_raw_hop_length` from their placeholder
  shape into the actual shape implied by `window_mode`/`hop_mode` (scalar,
  `[1, num_frames]`, `[freq_bins, 1]`, or `[freq_bins, num_frames]` for the
  window; scalar or `[num_frames]` for the hop), always re-wrapped as a new
  `nn.Parameter` with the correct `requires_grad`.
- Finishes with `self._validate_parameter_shapes()` (a private consistency
  check — every learnable tensor's shape must be broadcastable against
  `freq_bins`/`num_frames`) and sets `self._initialized = True`.

The whole method is decorated `@torch.no_grad()` — this setup step (shape
allocation, parameter re-wrapping) is not itself something to backpropagate
through.
