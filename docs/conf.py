# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "DSTFT"
author = "Maxime Leiber"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.imgmath",
    "sphinx_rtd_theme",
    "myst_parser",
    "nbsphinx",
    "sphinx-prompt",
    "sphinx_copybutton",
]
source_suffix = [".rst", ".md"]


templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autodoc_typehints = "description"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
nbsphinx_execute = "always"

latex_engine = "xelatex"
latex_theme = "manual"
latex_theme_options = {}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
