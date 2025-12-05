import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "qFabric"
primary_domain = "py"
author = "Mingyu Fan"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["qfabric.programmer.driver.m4i6622.pyspcm"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_static_path = ["_static"]
