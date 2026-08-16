"""Generate golden reference files for tests/test_golden.py.

Run manually (`python tests/golden/_generate_references.py`) only when a
change to src/dstft/ is an intentional behavior change and the golden
references need to be regenerated to match. Not part of the test suite
itself (excluded by `python_files = ["test_*.py"]`).
"""

from __future__ import annotations

import torch
from _configs import CONFIGS, golden_path, make_signal

from dstft import DSTFT


def main() -> None:
    for config in CONFIGS:
        x = make_signal(batch=config["batch"], length=config["length"])
        model = DSTFT(**config["kwargs"])
        model.initialize(x)
        model.eval()
        with torch.no_grad():
            spec, stft = model(x)
            reference = {"spec": spec, "stft": stft}
            if config.get("test_inverse", True):
                reference["x_hat"] = model.inverse(stft)

        torch.save(reference, golden_path(config["name"]))
        print(f"wrote {golden_path(config['name'])}")


if __name__ == "__main__":
    main()
