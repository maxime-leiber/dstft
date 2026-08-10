DSTFT
=====

.. image:: https://img.shields.io/pypi/v/dstft.svg
   :target: https://pypi.org/project/dstft/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/dstft.svg
   :target: https://pypi.org/project/dstft/
   :alt: Python versions

.. image:: https://readthedocs.org/projects/dstft/badge/?version=latest
   :target: https://dstft.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/maxime-leiber/dstft/actions/workflows/ci.yml
   :alt: CI

.. image:: https://codecov.io/gh/maxime-leiber/dstft/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/maxime-leiber/dstft
   :alt: Coverage

.. image:: https://img.shields.io/github/license/maxime-leiber/dstft.svg
   :target: https://github.com/maxime-leiber/dstft/blob/main/LICENSE
   :alt: License

**dstft** implements a **Differentiable Short-Time Fourier Transform**: a
PyTorch ``nn.Module`` (``DSTFT``) that computes a spectrogram conceptually
like ``torch.stft``, except every parameter that normally has to be fixed up
front — the analysis window's length, and the spacing between frames — can
instead be a learnable tensor. Because the whole transform is implemented
with differentiable PyTorch ops, gradients flow from a downstream loss (e.g.
"make this spectrogram sparse") back into the window length or hop length,
letting an optimizer discover a time-frequency tiling adapted to the signal
instead of a hand-picked one.

Unlike ``torch.stft``, ``DSTFT`` is initialized once per signal length
(``dstft.initialize(x)``) and returns both the magnitude spectrogram and the
complex transform (``spec, stft = dstft(x)``) — see :doc:`getting_started`
for a full example. Each instance is tied to the signal length it was first
initialized with; reinitializing with a different length raises
``RuntimeError`` (create a new instance instead).

.. image:: _static/opt.gif
   :alt: Optimization demo
   :width: 600

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   installation
   citation
   api
   notebooks
   internals
