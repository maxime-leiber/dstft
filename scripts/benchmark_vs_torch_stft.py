"""Micro-benchmark: DSTFT vs. torch.stft, CPU, forward speed and memory.

Compares DSTFT (window_mode="fixed", hop_mode="fixed" — its cheapest,
non-learnable configuration, closest to what torch.stft computes) against
torch.stft with equivalent parameters (same n_fft, hop_length, Hann window).
Also times a DSTFT forward+backward pass (window_mode="constant", learnable
win_length) as a reference for the actual training-loop cost, since that's
what torch.stft cannot do at all.

Run with: python scripts/benchmark_vs_torch_stft.py
"""

from __future__ import annotations

import torch
import torch.utils.benchmark as torch_benchmark
from torch.profiler import ProfilerActivity, profile

from dstft import DSTFT


N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 8
SIGNAL_LENGTHS = {
    "1s @16kHz (16,000 samples)": 16_000,
    "5s @16kHz (80,000 samples)": 80_000,
    "30s @16kHz (480,000 samples)": 480_000,
}


def torch_stft_forward(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
        center=True,
    )


def make_dstft(x: torch.Tensor, *, learnable: bool) -> DSTFT:
    model = DSTFT(
        n_fft=N_FFT,
        hop_length=float(HOP_LENGTH),
        win_length=float(N_FFT),
        window_mode="constant" if learnable else "fixed",
        hop_mode="fixed",
    )
    model.initialize(x)
    return model


def run_timing() -> None:
    header = (
        f"{'signal length':32s} {'torch.stft (ms)':>16s} "
        f"{'DSTFT fwd (ms)':>16s} {'DSTFT fwd+bwd (ms)':>20s}"
    )
    print(header)
    print("-" * len(header))
    for label, sig_len in SIGNAL_LENGTHS.items():
        x = torch.randn(BATCH_SIZE, sig_len)
        hann = torch.hann_window(N_FFT)

        t_stft = torch_benchmark.Timer(
            stmt="torch_stft_forward(x, hann)",
            globals={"torch_stft_forward": torch_stft_forward, "x": x, "hann": hann},
        ).blocked_autorange(min_run_time=1.0)

        dstft_fixed = make_dstft(x, learnable=False)
        with torch.no_grad():
            t_dstft_fwd = torch_benchmark.Timer(
                stmt="dstft_fixed(x)",
                globals={"dstft_fixed": dstft_fixed, "x": x},
            ).blocked_autorange(min_run_time=1.0)

        dstft_learnable = make_dstft(x, learnable=True)

        def fwd_bwd(
            dstft_learnable: DSTFT = dstft_learnable, x: torch.Tensor = x
        ) -> None:
            spec, _ = dstft_learnable(x)
            spec.sum().backward()
            dstft_learnable._raw_win_length.grad = None

        t_dstft_fwd_bwd = torch_benchmark.Timer(
            stmt="fwd_bwd()",
            globals={"fwd_bwd": fwd_bwd},
        ).blocked_autorange(min_run_time=1.0)

        print(
            f"{label:32s} {t_stft.mean * 1e3:16.2f} "
            f"{t_dstft_fwd.mean * 1e3:16.2f} {t_dstft_fwd_bwd.mean * 1e3:20.2f}"
        )


def run_memory() -> None:
    # Representative mid-size signal; memory profiling is more about relative
    # order of magnitude than a number that needs many signal lengths.
    x = torch.randn(BATCH_SIZE, SIGNAL_LENGTHS["5s @16kHz (80,000 samples)"])
    hann = torch.hann_window(N_FFT)
    dstft_fixed = make_dstft(x, learnable=False)

    print()
    print(
        "Forward-pass CPU memory (sum of positive per-op allocations, torch.profiler):"
    )
    for name, fn in [
        ("torch.stft", lambda: torch_stft_forward(x, hann)),
        ("DSTFT (fixed mode)", lambda: dstft_fixed(x)),
    ]:
        with (
            torch.no_grad(),
            profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof,
        ):
            fn()
        total_kib = (
            sum(
                evt.cpu_memory_usage
                for evt in prof.key_averages()
                if evt.cpu_memory_usage > 0
            )
            / 1024
        )
        print(f"  {name:22s} {total_kib:10.1f} KiB")


if __name__ == "__main__":
    print(
        f"PyTorch {torch.__version__}, CPU, batch_size={BATCH_SIZE}, "
        f"n_fft={N_FFT}, hop_length={HOP_LENGTH}\n"
    )
    run_timing()
    run_memory()
