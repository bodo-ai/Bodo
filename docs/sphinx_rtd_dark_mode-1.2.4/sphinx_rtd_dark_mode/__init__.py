"""
Sphinx Read the Docs Dark Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allows for toggable dark mode on the Read the Docs
theme for Sphinx.
"""

__title__ = "sphinx-rtd-dark-mode"
__description__ = "Dark mode for the Sphinx Read the Docs theme."
__author__ = "MrDogeBro"
__version__ = "1.2.4"
__license__ = "MIT"

from sphinx_rtd_dark_mode.dark_mode_loader import DarkModeLoader


def setup(app):
    app.add_config_value("default_dark_mode", True, "html")

    app.connect("config-inited", DarkModeLoader().configure)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
