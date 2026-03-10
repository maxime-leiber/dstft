# DSTFT

[![PyPI Version](https://img.shields.io/badge/pypi-v3.0.0-blue.svg)](https://pypi.org/project/dstft/)
[![Documentation Status](https://readthedocs.org/projects/dstft/badge/?version=latest)](https://dstft.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml/badge.svg)](https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml)
[![IEEE TSP](https://img.shields.io/badge/IEEE_TSP-DSTFT-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11220928)

**DSTFT** (Differentiable Short-Time Fourier Transform) is a PyTorch module for a differentiable short-time Fourier transform, supporting learnable/adaptive parameters.

---

<!-- For GitHub -->
<img src="docs/_static/opt.gif" alt="Optimization demo" width="600"/>

Gradient-based optimization of DSTFT parameters (example: window length).

---

## Features

- Differentiable STFT (learnable window lengths, and hop lengths)
- FFT for DSTFT, DFT for adaptive DSTFT, inverse DSTFT

## Installation

### pip/venv

Create and activate a virtual environment, then install in editable mode:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

### Conda/Mamba + uv

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


## License

This project is licensed under the terms of the MIT License. See the
[LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes,
improvements, or new features.

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
