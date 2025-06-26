# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../gwsnr")) 

# Import version dynamically
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "..", "gwsnr", "_version.py")
    version_dict = {}
    with open(version_file) as f:
        exec(f.read(), version_dict)
    return version_dict['__version__']

project = 'gwsnr'
copyright = '2023, Phurailatpam Hemantakumar'
author = 'Phurailatpam Hemantakumar'
release = get_version()


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.inheritance_diagram",
    "sphinx_tabs.tabs",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints','.ipynb', "venv", ".*", '**.ipynb_checkpoints', '**/*.ipynb_checkpoints', '**/**/*.ipynb_checkpoints', '**/**/**/*.ipynb_checkpoints']
autodoc_member_order = 'bysource'
numpydoc_show_class_members = False
autoapi_add_toctree_entry = False
# -- Napoleon options
napoleon_include_special_with_doc = True
pygments_style = 'sphinx'

# Don't mess with double-dash used in CLI options
smartquotes_action = "qe"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# -- Plausible support
ENABLE_PLAUSIBLE = os.environ.get("READTHEDOCS_VERSION_TYPE", "") in ["branch", "tag"]
html_context = {"enable_plausible": ENABLE_PLAUSIBLE}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',  # Add this line to include your custom CSS
]

# -- Configure autoapi -------------------------------------------------------
autodoc_typehints = "signature"  # autoapi respects this

autoapi_type = "python"
autoapi_dirs = ["../gwsnr"]
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "special-members",
]
# autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = True
# autoapi_generate_api_docs = False

# -- Napoleon options
napoleon_include_special_with_doc = True

def skip_member(app, what, name, obj, skip, options):
    if what == "gwsnr.C":
        skip = True
        
    if "gwsnr" in name:
        if obj.name in [
            "C",
        ]:
            skip = True
    return skip

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_member)
    roles.register_local_role('orange', orange_role)
    roles.register_local_role('orange_first', orange_first_letter)
    roles.register_local_role('red', red_role)
    roles.register_local_role('red_first', red_first_letter)

from docutils import nodes
from docutils.parsers.rst import roles

def orange_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=["orange"])
    return [node], []

# def setup(app):
#     roles.register_local_role('orange', orange_role)

def orange_first_letter(role, rawtext, text, lineno, inliner, options={}, content=[]):
    # Create two nodes: one for the first letter with a class and one for the rest
    first_letter = nodes.inline(rawtext, text[0], classes=["orange"])
    rest = nodes.inline(rawtext, text[1:], classes=[])
    return [first_letter, rest], []

# for red
def red_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=["red"])
    return [node], []

def red_first_letter(role, rawtext, text, lineno, inliner, options={}, content=[]):
    # Create two nodes: one for the first letter with a class and one for the rest
    first_letter = nodes.inline(rawtext, text[0], classes=["red"])
    rest = nodes.inline(rawtext, text[1:], classes=[])
    return [first_letter, rest], []

# def setup(app):
#     roles.register_local_role('orange', orange_role)
#     roles.register_local_role('orange_first', orange_first_letter)
#     roles.register_local_role('red', red_role)
#     roles.register_local_role('red_first', red_first_letter)