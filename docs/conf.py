# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("..")) 

project = 'gwsnr'
copyright = '2023, Phurailatpam Hemantakumar'
author = 'Phurailatpam Hemantakumar'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx_tabs.tabs",
    "autoapi.extension",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints','.ipynb']
numpydoc_show_class_members = False
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme = "sphinx_rtd_theme"

# -- Configure autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../gwsnr/"]
autoapi_add_toctree_entry = False
autoapi_options = ["members", "show-inheritance", "show-module-summary"]
