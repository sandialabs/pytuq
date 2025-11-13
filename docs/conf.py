# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# find project
sys.path.insert(0, os.path.abspath('../src'))

# from sphinx_gallery.sorting import FileNameSortKey
import warnings
import pkg_resources

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyTUQ'
year = '2025'
author = "Bert Debusschere, Khachik Sargsyan, Emilie Baillo"
copyright = f"{year}, {author}"

source_suffix = ".rst"
master_doc = "index"

from pkg_resources import get_distribution, DistributionNotFound

try:
    version = get_distribution("package-name").version
except DistributionNotFound:
    version = release = "1.0.0"
    pass

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    # "numpydoc",
    "myst_parser"
]

extensions += ['matplotlib.sphinxext.plot_directive',
               'IPython.sphinxext.ipython_console_highlighting',
               'IPython.sphinxext.ipython_directive']

# extensions += ['sphinx_gallery.gen_gallery']

numpydoc_show_class_members = False 

# Bibliography configuration
bibtex_bibfiles = ['references.bib']

# -----------------------------------------------------------------------------
# Sphinx AutoAPI
# -----------------------------------------------------------------------------

extensions += ['autoapi.extension']
autoapi_dirs = ['../src'] # Where the PyTUQ source code is
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi" # Templates for AutoAPI documentation
# autoapi_add_toctree_entry = False  # Adding the generateed documentation into the TOC Tree
suppress_warnings = ["autoapi", "ref.python"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary"
]

autoapi_own_page_level = 'module'
autoapi_keep_files = True # Keep the AutoAPI generated files on the filesystem

# -----------------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

source_suffix = ".rst"
master_doc = "index"
pygments_style = "trac"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'flyout_display': 'hidden',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/', None),
}
