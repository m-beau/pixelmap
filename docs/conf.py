# Configuration file for the Sphinx documentation builder.

import pixelmap

project = "PixelMap"
copyright = "2024, Maxime Beau"
author = "Maxime Beau"
version = pixelmap.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Napoleon settings
napoleon_google_style = True
napoleon_numpy_style = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "PixelMap"
html_logo = "_static/npix_map_logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/m-beau/pixelmap",
    "source_branch": "main",
    "source_directory": "docs/",
}
