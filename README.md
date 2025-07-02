# DSTFT

![opt gif](resources/opt.gif)

[![Tests](https://github.com/USERNAME/DSTFT/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/DSTFT/actions)
[![Documentation Status](https://readthedocs.org/projects/dstft/badge/?version=latest)](https://dstft.readthedocs.io/en/latest/?badge=latest)

**DSTFT** (Differentiable Short-Time Fourier Transform) is a PyTorch module for differentiable short-time Fourier transform, supporting adaptive windows and strides.

### Install DSTFT package : create / update conda environment

#### via mamba / conda and uv pip (recommended)

Create env
```bash
mamba create -n dstft python=3.11 pip=20.2 uv 
mamba activate dstft
```

Install dependencies
```bash
uv pip install -e .
```

### You can of course install without conda and without uv

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

```python
from dstft.dstft import DSTFT
from dstft.adstft import ADSTFT
dstft = DSTFT(
    x,
    win_length=...,
    support=...,
    stride=...,
    win_requires_grad=...,
    ...
)
spec, stft = dstft(x)
dstft.plot(spec)
```

## Documentation

Full documentation is generated with Sphinx:

## Examples

See the `notebooks/` folder for advanced use cases (speech, tracking, etc).

## Resources

The `resources/` folder contains images, gifs, and useful documents for the project.

## Citation

Please cite this repository if you use it in your scientific work.

```bibtex
@article{leiber2025learnable,
  title={Learnable Adaptive Time-Frequency Representation via Differentiable Short-Time Fourier Transform},
  author={Leiber, Maxime and Marnissi, Yosra and Barrau, Axel and Meignen, Sylvain and Massouli{\~A}{\v{S}}, Laurent},
  journal={arXiv preprint arXiv:2506.21440},
  year={2025}
}

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
'''

[![Paper1](http://img.shields.io/badge/paper1-arxiv-b31b1b.svg)](https://arxiv.org/pdf/2506.21440) [![Paper2](http://img.shields.io/badge/paper1-arxiv-b31b1b.svg)](https://arxiv.org/abs/2208.10886) [![Paper3](http://img.shields.io/badge/paper2-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02418) [![Paper4](http://img.shields.io/badge/paper3-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02421)