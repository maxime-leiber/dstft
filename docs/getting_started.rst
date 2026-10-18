Getting started
===============

Install the package:

.. code-block:: bash

   pip install dstft

See :doc:`installation` for other installation methods (``uv``, Conda/Mamba,
or an editable install for development).

Quick example:

.. code-block:: python

   import torch
   from dstft import DSTFT

   torch.manual_seed(0)
   x = torch.randn(1, 1024)

   dstft = DSTFT(n_fft=256, hop_length=64.0, win_length=256.0, window_mode="constant")
   dstft.initialize(x)
   spec, stft = dstft(x)
