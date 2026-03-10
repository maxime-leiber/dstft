# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# Avoid importing a different `dstft` package from the environment.
sys.modules.pop("dstft", None)

# Ensure notebooks executed by nbsphinx import the local sources.
nbsphinx_prolog = r"""
.. raw:: html

    <script type="text/javascript">
    (function() {
      // Make sure executed notebooks import the local checkout.
      // nbsphinx runs in the docs/ directory, so the package sources are in ../src.
      if (typeof window !== 'undefined') {
        // no-op for browser
      }
    })();
    </script>

.. code-block:: python

    import os, sys
    sys.path.insert(0, os.path.abspath('../src'))
"""

project = "DSTFT"
author = "Maxime Leiber"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

autosummary_generate = True
autodoc_typehints = "description"

html_theme = "alabaster"
html_static_path = ["_static"]

# Source formats
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

# Allow toggling notebook execution via env: set NBS_EXEC=1 to execute.
NBS_EXEC = os.environ.get("NBS_EXEC", "0")
nbsphinx_execute = "auto" if NBS_EXEC == "1" else "never"
nbsphinx_allow_errors = False

if os.environ.get("SPHINX_OFFLINE"):
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "torch": ("https://pytorch.org/docs/stable", None),
    }
