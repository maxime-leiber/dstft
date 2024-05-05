# DSTFT for an adaptive and learnable time-frequency representation

This repository contains the code of differentiable short-time Fourier transform (DSTFT). It allows automatic tuning of spectrogram parameters by optimizing them by gradient descent. DSTFT is a time-frequency neural network layer whose weights are its parameters (i.e. window and hop lengths at the moment).

![alt text](fig/opt.gif "Logo Title Text 1")

## Installation

### Clone sphm Package

```bash
git clone https://github.com/maxime-leiber/dstft
cd dstft
```

### Install dstft Package

#### create new conda env

```bash
conda env create -f environment.yml
```

#### or update existing conda env

```bash
conda env update -f environment.yml
```

#### or use pip

```bash
pip install -e .
```




<p float="middle">
  <img src="fig/1_100.png" width="15%" />
  <img src="fig/1_1000.png" width="15%" /> 
  <img src="fig/1_1.png" width="15%" />
  <img src="fig/1_2.png" width="15%" /> 
  <img src="fig/1_3.png" width="15%" />
  <img src="fig/1_4.png" width="15%" /> 
</p>

<p float="middle">
  <img src="fig/2_100.png" width="15%" />
  <img src="fig/2_1000.png" width="15%" /> 
  <img src="fig/2_1.png" width="15%" />
  <img src="fig/2_2.png" width="15%" /> 
  <img src="fig/2_3.png" width="15%" />
  <img src="fig/2_4.png" width="15%" /> 
</p>

<p float="middle">
  <img src="fig/5_.png" width="15%" />
  <img src="fig/5_1.png" width="15%" /> 
  <img src="fig/5_2.png" width="15%" />
</p>

<p float="middle">
  <img src="fig/3_100.png" width="15%" />
  <img src="fig/3_100b.png" width="15%" /> 
  <img src="fig/3_1000.png" width="15%" />
  <img src="fig/3_1000b.png" width="15%" /> 
  <img src="fig/3_3.png" width="15%" />
  <img src="fig/3_4.png" width="15%" /> 
</p>

<p float="middle">
  <img src="fig/4_100.png" width="15%" />
  <img src="fig/4_100b.png" width="15%" /> 
  <img src="fig/4_1000.png" width="15%" />
  <img src="fig/4_1000b.png" width="15%" /> 
  <img src="fig/4_3.png" width="15%" />
  <img src="fig/4_4.png" width="15%" /> 
</p>


[![Paper1](http://img.shields.io/badge/paper1-arxiv-b31b1b.svg)](https://arxiv.org/abs/2208.10886) [![Paper2](http://img.shields.io/badge/paper2-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02418) [![Paper3](http://img.shields.io/badge/paper3-arxiv-b31b1b.svg)](https://arxiv.org/abs/2308.02421)
