# DSTFT

[![PyPI Version](https://img.shields.io/pypi/v/dstft.svg)](https://pypi.org/project/dstft/)
[![Python Versions](https://img.shields.io/pypi/pyversions/dstft.svg)](https://pypi.org/project/dstft/)
[![Documentation Status](https://readthedocs.org/projects/dstft/badge/?version=latest)](https://dstft.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml/badge.svg)](https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maxime-leiber/dstft/branch/main/graph/badge.svg)](https://codecov.io/gh/maxime-leiber/dstft)
[![License](https://img.shields.io/github/license/maxime-leiber/dstft.svg)](LICENSE)
[![IEEE TSP](https://img.shields.io/badge/IEEE_TSP-DSTFT-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11220928)

**dstft** implements a **Differentiable Short-Time Fourier Transform**: a PyTorch `nn.Module` (`DSTFT`) that computes a spectrogram conceptually like `torch.stft`, except every parameter that normally has to be fixed up front — the analysis window's length, and the spacing between frames — can instead be a learnable tensor. Because the whole transform is implemented with differentiable PyTorch ops, gradients flow from a downstream loss (e.g. "make this spectrogram sparse") back into the window length or hop length, letting an optimizer discover a time-frequency tiling adapted to the signal instead of a hand-picked one.

Unlike `torch.stft`, `DSTFT` is initialized once per signal length (`dstft.initialize(x)`) and returns both the magnitude spectrogram and the complex transform (`spec, stft = dstft(x)`) — see the usage example below. Each instance is tied to the signal length it was first initialized with; reinitializing with a different length raises `RuntimeError` (create a new instance instead).

---

<!-- For GitHub -->
<img src="docs/_static/opt.gif" alt="Optimization demo" width="600"/>

Gradient-based optimization of DSTFT parameters (example: window length).

---

## Features

- Differentiable STFT (learnable window lengths, and hop lengths)
- FFT for DSTFT, DFT for adaptive DSTFT, inverse DSTFT

## Installation

### From PyPI

For general use, install the published package:

```bash
pip install dstft
```

Or with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install dstft
```

Or in a Conda/Mamba environment (there is no separate conda-forge package;
`pip install` works the same once the environment is activated):

```bash
mamba create -n dstft python=3.11 pip
mamba activate dstft
pip install dstft
```

### For development (editable install)

To contribute to `dstft` itself, clone the repository and install in
editable mode instead — see [Contributing](#contributing) below.

#### pip/venv

Create and activate a virtual environment, then install in editable mode:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

#### Conda/Mamba + uv

Create a new environment:

```bash
mamba create -n dstft python=3.11 pip
mamba activate dstft
pip install -U uv
```

Install the package:

```bash
uv pip install -e .
```

Install optional dependencies:

```bash
uv pip install -e ".[dev,docs]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

For documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Usage example

```python
import torch

from dstft import DSTFT

torch.manual_seed(0)
x = torch.randn(1, 1024)

dstft = DSTFT(
    n_fft=256,
    hop_length=64.0,
    win_length=256.0,
    window_mode="constant",
)
dstft.initialize(x)

spec, stft = dstft(x)
```

Since `window_mode="constant"` makes `win_length` a learnable parameter, it
can be optimized directly by gradient descent — no separate API, just a
normal PyTorch training loop against any loss defined on `spec`/`stft`:

```python
optimizer = torch.optim.Adam(dstft.parameters(), lr=1.0)
for _ in range(200):
    optimizer.zero_grad()
    spec, _ = dstft(x)
    loss = spec.sum(dim=(1, 2)).mean()  # e.g. any loss defined on the spectrogram
    loss.backward()
    optimizer.step()

print(dstft.win_length)  # moved away from its initial value of 256.0
```

See `notebooks/inverse.ipynb` for a full worked example, including
reconstructing the signal back from the transform.

## License

This project is licensed under the terms of the MIT License. See the
[LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes,
improvements, or new features.

## Releasing

The package version is derived automatically from git tags (via
`setuptools-scm`) — it is never edited by hand in `pyproject.toml`. To cut a
release:

```bash
git tag v3.0.1
git push origin v3.0.1
```

Then build and publish as usual (`python -m build && twine upload dist/*`).
A tag on a commit with no uncommitted local changes produces an exact version
(`3.0.1`); any other commit gets an automatic `.devN+g<hash>` suffix.

## Citation

Please cite this repository if you use it in your scientific work:

```bibtex
@ARTICLE{11220928,
  author={Leiber, Maxime and Marnissi, Yosra and Barrau, Axel and Meignen, Sylvain and Massoulié, Laurent},
  journal={IEEE Transactions on Signal Processing},
  title={Optimal Adaptive Time-Frequency Representation via Differentiable Short-Time Fourier Transform},
  year={2025},
  volume={73},
  number={},
  pages={5047-5059},
  keywords={Windows;Time-frequency analysis;Optimization;Spectrogram;Computational efficiency;Tuning;Signal resolution;Neural networks;Discrete Fourier transforms;Backpropagation;Short-time Fourier transform;spectrogram;differentiable STFT;learnable STFT parameters;adaptive time-frequency representation},
  doi={10.1109/TSP.2025.3624477}}
@inproceedings{leiber2022differentiable,
  title={A differentiable short-time Fourier transform with respect to the window length},
  author={Leiber, Maxime and Barrau, Axel and Marnissi, Yosra and Abboud, Dany},
  booktitle={2022 30th European Signal Processing Conference (EUSIPCO)},
  pages={1392--1396},
  year={2022},
  organization={IEEE}
}
@inproceedings{leiber2023differentiable,
  title={Differentiable adaptive short-time Fourier transform with respect to the window length},
  author={Leiber, Maxime and Marnissi, Yosra and Barrau, Axel and El Badaoui, Mohammed},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
@inproceedings{leiber2023differentiable,
  title={Differentiable short-time Fourier transform with respect to the hop length},
  author={Leiber, Maxime and Marnissi, Yosra and Barrau, Axel and El Badaoui, Mohammed},
  booktitle={2023 IEEE Statistical Signal Processing Workshop (SSP)},
  pages={230--234},
  year={2023},
  organization={IEEE}
}
```

[![IEEE TSP](https://img.shields.io/badge/IEEE_TSP-DSTFT-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11220928)
[![EUSIPCO](https://img.shields.io/badge/EUSIPCO-2208.10886-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2208.10886)
[![ICASSP](https://img.shields.io/badge/ICASSP-2506.21440-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2308.02418)
[![SSP Workshop](https://img.shields.io/badge/SSP_Workshop-2308.02418-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2308.02421)
