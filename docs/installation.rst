Installation
============

Universal (pip/venv)
--------------------

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
------------------------------

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

