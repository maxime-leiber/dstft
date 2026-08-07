# `dstft/_core.py`

The math kernels behind `DSTFT`. Everything here is a free function operating
on plain tensors — no class, no stored state, nothing hidden. `dstft.py`
holds the configuration and validation; this file is where the actual
transform is computed. Two small module-level caches
(`_FFT_FORWARD_CACHE`, `_DFT_TWIDDLE_CACHE`, keyed by shape/device/dtype)
memoize tensors that are expensive to rebuild but never change for a given
configuration (a bin-index arange, and a DFT twiddle-factor matrix).

## The floor/frac convention (the idea everything else builds on)

A recurring pattern through this whole file: a **frame center** `t_n` can be
any real number, not just an integer sample index. Every function that needs
one splits it into `idx_floor = floor(t_n)` (used for actual array indexing —
you can't index an array at a fractional position) and `idx_frac = t_n -
idx_floor` (used downstream: to continuously shift the analysis window in
`windows.hann_window`, and to apply a frequency-domain phase-correction
factor so the *effective* window is still centered at the true `t_n`, not
just at `idx_floor`). This split is what the module docstrings across this
file refer to as "paper Eq. (25)".

## Frame positions and extraction

### `compute_frame_positions_fixed_hop(*, signal_length, n_fft, hop_length, device, dtype) -> Tensor[frames]`

Computes evenly-spaced frame centers for the common case of a constant hop:
`t_n = n * hop_length` for `n = 0, 1, ..., num_frames - 1`, where
`num_frames = 1 + floor(signal_length / hop_length)` (the `+1` and `floor`
together are what the code calls the "cover" policy — enough frames to span
the whole signal, even if the last one or two extend past its end, at which
point they're implicitly zero-padded rather than clipped). `hop_length` can
be a plain float or itself a tensor (so this composes with a learnable
scalar hop length), but the *count* of frames is computed from a detached
value — the number of frames is a **shape**, and shapes can't depend on a
tensor that requires gradients.

### `unfold_floor_frac(*, x, frame_positions, n_fft) -> (frames, idx_floor, idx_frac)`

Given a batch of signals `x: [batch, time]` and a set of frame centers,
extracts one length-`n_fft` window of raw samples around each center:
`k ∈ {-n_fft//2, ..., n_fft//2 - 1}` relative to `floor(t_n)`. Any sample
index that falls outside `[0, signal_length)` — i.e. a frame near the
start/end of the signal — is replaced with `0` via a boolean mask rather than
actually padding the input tensor (`torch.where(valid, gathered, 0)`), which
is what makes the "implicit zero-padding" mentioned throughout this file
implicit: there is no padded copy of `x` anywhere, just a validity mask
applied at gather time.

## Forward transform: two backends

### `fdstft_fft_forward(*, frames, analysis_window, n_fft, idx_frac, idx_floor, fft_norm=None) -> stft[batch, freq, frames]`

The fast path, used whenever the window is at most time-varying (not
frequency-varying). Multiplies each extracted frame by its window
(broadcastable — the same window can be shared across all frames, or vary
per frame), runs a single batched `torch.fft.rfft` over all
`batch * num_frames` windowed frames at once, then applies the phase
correction factor `exp(-j·2π·floor(t_n)·m / n_fft)` per frequency bin `m` to
account for the frame having actually been centered at the fractional `t_n`
and not at `floor(t_n)`. Because this is a real FFT of a real input, one
`rfft` call handles every frame — this is the reason a *frequency-varying*
window can't use this path: `rfft` can't apply a different window per output
bin.

### `adstft_dft_forward(*, frames, analysis_window, n_fft, idx_frac, idx_floor) -> stft[batch, freq_bins, frames]`

The general path, used when the window varies across frequency (`"frequency"`
or `"time-frequency"` mode). Since each frequency bin can have its own
window, it's no longer possible to reuse a single FFT across bins, so this
function instead: (1) multiplies frames by a *per-frequency-bin* window
tensor `analysis_window: [freq_bins, frames, n_fft]` (broadcasting over
`freq_bins`/`frames` where those dims are `1`), producing a
`[batch, freq_bins, frames, n_fft]` tensor of tapered frames, then (2)
contracts against a DFT twiddle matrix with `torch.einsum("bftk,fk->bft", ...)`
— literally computing, for each `(batch, freq, frame)`, the DFT sum over the
local sample index `k` by hand instead of via `torch.fft`. Slower (an
explicit matrix contraction rather than an FFT), but it's the only way to let
frequency bin `m`'s DFT use a window that no other bin shares. The same
phase-correction step as the FFT backend is applied afterward.

Read together, `fdstft_fft_forward` and `adstft_dft_forward` implement the
*same* transform when their window happens to be frequency-independent —
`_core.py`'s comments call out that `adstft_dft_forward` follows the
`torch.stft` convention over the *local* frame index and therefore does
**not** need the extra phase factor baked into the twiddle matrix itself
(the phase correction is applied as a separate multiplication after the
`einsum`, mirroring the FFT path).

## Inverse: adjoint, WOLA, and CG

### `overlap_add_wola(*, frames, frame_positions, analysis_window, signal_length, eps) -> Tensor[batch, signal_length]`

A weighted overlap-add reconstruction that loops over frames in Python
(`for t in range(num_frames)`), scatter-adding each windowed frame into the
output signal and separately accumulating the sum of squared window values
per output sample, then dividing (`num / (den + eps)`). This is the
straightforward, unvectorized reference implementation; `overlap_add_dual`
below is the vectorized version actually used by `DSTFT.inverse`.

### `overlap_add_dual(*, frames, frame_positions, analysis_window, signal_length, eps) -> Tensor[batch, signal_length]`

The version `DSTFT.inverse()` actually calls for its FFT-backend path. Same
weighted-overlap-add idea as `overlap_add_wola`, but vectorized using
`index_add_` instead of a Python loop over frames — flattens the
`(frame, local-sample)` grid into 1D index/value arrays once, then does two
`index_add_` scatter-sums (one to build the denominator, one to build the
weighted numerator) instead of iterating. Functionally equivalent, much
faster on GPU. This is why `DSTFT.inverse()` achieves exact reconstruction
on the FFT backend: the numerator of this division is, by construction,
`x[p] * (sum of squared window values at p)`, which exactly cancels the
denominator wherever a sample is actually covered by at least one frame.

### `overlap_add_dual_dft_wola_den(*, analysis_window, n_fft, frame_positions, signal_length, eps) -> Tensor[signal_length]`

Computes just the denominator (not the full reconstruction) for the
DFT-backend inverse, accounting for the fact that a real-valued signal's
`rfft` only stores non-negative frequencies — each bin except DC (and
Nyquist, if present) actually represents *two* symmetric frequency
components, so it must be weighted by `2` (`c_m`) rather than `1` when
summing squared-window energy across frequency bins. This denominator is
reused both as the normalizer for the fast `"wola"` inverse method and as a
diagonal *preconditioner* for the `"cg"` method's conjugate-gradient solve.

### `adstft_dft_forward`'s adjoint — `adstft_dft_adjoint(*, stft, analysis_window, n_fft, frame_positions, signal_length) -> Tensor[batch, signal_length]`

The exact linear adjoint `A*` of `adstft_dft_forward`, i.e. the operation you
compose with a denominator (either the WOLA diagonal above, for a fast
approximate inverse, or as one step of a conjugate-gradient loop against
`(A*A + λI)`, for an accurate one). Undoes the forward phase-correction
factor (conjugated), applies the same rFFT multiplicity weights `c_m`, then
runs an einsum against the conjugate-transposed twiddle matrix and
scatter-adds the result back onto the time axis with a Python loop (this
function is explicitly the exact/reference adjoint, not the CUDA-fast path —
it trades vectorization for correctness/clarity where the two matter more,
since it's also used as the residual computation inside `cg_solve`).

### `cg_solve(*, apply_mat, b, x0=None, max_iter=200, tol=1e-10, precond=None) -> Tensor`

A generic, batched preconditioned conjugate-gradient solver for symmetric
positive-definite linear systems `apply_mat(x) = b`. Doesn't know anything
about DSTFT specifically — `apply_mat` is passed in as a closure by
`DSTFT.inverse()`'s `"cg"` branch, composed from `adstft_dft_forward` +
`adstft_dft_adjoint` + a `λI` regularization term (`apply_a` → `apply_at` →
`+ cg_lambda * x`), and `precond` is a diagonal preconditioner built from
`overlap_add_dual_dft_wola_den`. Standard PCG: iterates residual/search
direction updates until either the relative residual drops below `tol` or
`max_iter` is reached, guarding the step-size denominator against
near-zero values to avoid division blow-ups on nearly-singular systems.
