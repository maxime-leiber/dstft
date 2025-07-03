# DSTFT

[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/dstft/)
[![Python Version](https://img.shields.io/pypi/pyversions/dstft.svg)](https://pypi.org/project/dstft/)
[![Coverage Status](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](./coverage.xml)
[![License](https://img.shields.io/github/license/maxime-leiber/dstft.svg)](./LICENSE)
[![CI](https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml/badge.svg)](https://github.com/maxime-leiber/dstft/actions)
[![Documentation Status](https://readthedocs.org/projects/dstft/badge/?version=latest)](https://dstft.readthedocs.io/en/latest/?badge=latest)

**DSTFT** (Differentiable Short-Time Fourier Transform) is a PyTorch module for differentiable short-time Fourier transform, supporting adaptive windows and strides. It is designed for research and production, with high test coverage and modern Python best practices.

---

## Features
- Differentiable STFT and inverse STFT
- Adaptive (learnable) window and stride (ADSTFT)
- PyTorch-first API
- High test coverage (>95%)
- Extensive documentation and examples

## Installation

### Using Conda/Mamba and uv pip (recommended)

Create a new environment:
```bash
mamba create -n dstft python=3.11 pip=20.2 uv
mamba activate dstft
```

Install dependencies:
```bash
uv pip install -e .
```

### Using pip/venv (no conda required)

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage Example

```python
import torch
from dstft import DSTFT, ADSTFT, __version__

print("DSTFT version:", __version__)

# Example input: batch of 1, 1D signal of length 1024
torch.manual_seed(0)
x = torch.randn(1, 1024)

# Create a DSTFT instance
dstft = DSTFT(x, win_length=128, support=128, stride=32)

# Forward pass
spec, stft = dstft(x)

# Plot the spectrogram
dstft.plot(spec)
```

## Documentation

Full documentation is generated with Sphinx and available at: [https://dstft.readthedocs.io/en/latest/](https://dstft.readthedocs.io/en/latest/)

## Examples

See the `notebooks/` folder for advanced use cases (speech, tracking, etc).

## Running Tests

To run the test suite, use:
```bash
pytest tests/
```

## Resources

The `resources/` folder contains images, gifs, and useful documents for the project.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## Citation

Please cite this repository if you use it in your scientific work:

```bibtex
@article{flamary2020differentiable,
  title={Differentiable Short Time Fourier Transform for Time Series Data},
  author={Flamary, R{\'e}mi and others},
  journal={arXiv preprint arXiv:2002.10211},
  year={2020}
}
```

[![Paper1](http://img.shields.io/badge/paper1-arxiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21440)
[![Paper2](http://img.shields.io/badge/paper2-arxiv-b31b1b.svg)](https://arxiv.org/abs/2208.10886)
[![Paper3](http://img.shields.io/badge/paper3-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02418)
[![Paper4](http://img.shields.io-badge/paper4-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02421)

---

![opt gif](resources/opt.gif)