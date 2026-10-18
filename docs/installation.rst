Installation
============

From PyPI
---------

For general use, install the published package:

.. code-block:: bash

   pip install dstft

Or with `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install dstft

Or in a Conda/Mamba environment (there is no separate conda-forge package;
``pip install`` works the same once the environment is activated):

.. code-block:: bash

   mamba create -n dstft python=3.11 pip
   mamba activate dstft
   pip install dstft


For development (editable install)
-----------------------------------

To contribute to ``dstft`` itself, clone the repository and install in
editable mode instead.

Universal (pip/venv)
~~~~~~~~~~~~~~~~~~~~~

Create and activate a virtual environment, then install in editable mode:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   pip install -U pip
   pip install -e .

Optional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   pip install -e ".[docs]"


Conda/Mamba + uv (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new environment:

.. code-block:: bash

   mamba create -n dstft python=3.11 pip
   mamba activate dstft
   pip install -U uv

Install the package:

.. code-block:: bash

   uv pip install -e .

Install optional dependencies:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"
