"""Golden-output numerical regression tests.

Compares DSTFT.forward()/inverse() against reference tensors frozen in
tests/golden/*.pt (see tests/golden/_generate_references.py). Catches
silent numerical regressions in src/dstft/ -- including ones a
"no behavior change" refactor could introduce by mistake -- that a purely
shape/finiteness-based test would miss.
"""

from __future__ import annotations

import pytest
import torch
from tests.golden._configs import CONFIGS, golden_path, make_signal

from dstft import DSTFT


# Loose enough to tolerate cross-platform/cross-torch-version floating-point
# differences in FFT/BLAS implementations, tight enough to catch an actual
# regression in dstft's own math (which would show up as a much larger
# deviation, not a last-bit rounding difference).
RTOL = 1e-4
ATOL = 1e-6


def _run(config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = make_signal(batch=config["batch"], length=config["length"])
    model = DSTFT(**config["kwargs"])
    model.initialize(x)
    model.eval()
    with torch.no_grad():
        spec, stft = model(x)
        x_hat = model.inverse(stft) if config.get("test_inverse", True) else None
    return spec, stft, x_hat


def test_golden_references_exist() -> None:
    for config in CONFIGS:
        assert golden_path(config["name"]).exists(), (
            f"missing golden reference for {config['name']!r}; run "
            "tests/golden/_generate_references.py"
        )


def _ids(configs: list[dict]) -> list[str]:
    return [c["name"] for c in configs]


@pytest.mark.parametrize("config", CONFIGS, ids=_ids(CONFIGS))
def test_forward_matches_golden_reference(config: dict) -> None:
    spec, stft, x_hat = _run(config)
    reference = torch.load(golden_path(config["name"]), weights_only=True)

    torch.testing.assert_close(spec, reference["spec"], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(stft, reference["stft"], rtol=RTOL, atol=ATOL)

    if config.get("test_inverse", True):
        assert x_hat is not None
        torch.testing.assert_close(x_hat, reference["x_hat"], rtol=RTOL, atol=ATOL)
    else:
        assert "x_hat" not in reference
