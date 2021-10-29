import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx_rtd_dark_mode",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
