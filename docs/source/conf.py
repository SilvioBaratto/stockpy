# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import sphinx_rtd_theme

# import pkg_resources

# -*- coding: utf-8 -*-
#
# Pyro documentation build configuration file, created by
# sphinx-quickstart on Thu Jun 15 17:16:14 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",  #
    "sphinx.ext.todo",  #
    "sphinx.ext.mathjax",  #
    "sphinx.ext.ifconfig",  #
    "sphinx.ext.viewcode",  #
    "sphinx.ext.githubpages",  #
    "sphinx.ext.graphviz",  #
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    'sphinx.ext.napoleon',
]

# Disable documentation inheritance so as to avoid inheriting
# docstrings in a different format, e.g. when the parent class
# is a PyTorch class.

autodoc_inherit_docstrings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"Pyro"
copyright = u"2017-2018, Uber Technologies, Inc"
author = u"Uber AI Labs"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = ""

if "READTHEDOCS" not in os.environ:
    # if developing locally, use pyro.__version__ as version
    from pyro import __version__  # noqaE402

    version = __version__

# release version
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# do not prepend module name to functions
add_module_names = False

# -- Options for HTML output ----------------------------------------------

# logo
html_logo = "_static/img/logo.png"

# logo
# html_favicon = "_static/img/favicon/favicon.ico"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "navigation_depth": 3,
    "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_style = "css/stockpy.css"

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "StockpyDoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "StockPy.tex", u"StockPy Documentation", u"Silvio Baratto", "manual"),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "StockPy", u"Stockpy Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "StockPy",
        u"StockPy Documentation",
        author,
        "StockPy",
        "Deep Learning Library",
        "Miscellaneous",
    ),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "funsor": ("http://funsor.pyro.ai/en/stable/", None),
    "opt_einsum": ("https://optimized-einsum.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "Bio": ("https://biopython.org/docs/latest/api/", None),
    "horovod": ("https://horovod.readthedocs.io/en/stable/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
}

# document class constructors (__init__ methods):
""" comment out this functionality for now;
def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip
"""


def setup(app):
    app.add_css_file("css/stockpy.css")


#     app.connect("autodoc-skip-member", skip)


# @jpchen's hack to get rtd builder to install latest pytorch
# See similar line in the install section of .travis.yml
if "READTHEDOCS" in os.environ:
    os.system("pip install numpy")
    os.system(
        "pip install torch==2.0+cpu torchvision==0.15.0+cpu "
        "-f https://download.pytorch.org/whl/torch_stable.html"
    )
