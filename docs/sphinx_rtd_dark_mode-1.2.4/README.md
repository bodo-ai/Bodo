# Sphinx RTD Theme Dark Mode

[![Build Status](https://travis-ci.com/MrDogeBro/sphinx_rtd_dark_mode.svg?branch=main)](https://travis-ci.com/github/MrDogeBro/content_filter) [![PyPi version](https://img.shields.io/pypi/v/sphinx-rtd-dark-mode.svg)](https://pypi.python.org/pypi/sphinx-rtd-dark-mode/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/sphinx-rtd-dark-mode.svg)](https://pypi.python.org/pypi/sphinx-rtd-dark-mode/) [![PyPI download month](https://img.shields.io/pypi/dm/sphinx-rtd-dark-mode.svg)](https://pypi.python.org/pypi/sphinx-rtd-dark-mode/) [![License](https://img.shields.io/github/license/MrDogeBro/sphinx_rtd_dark_mode.svg)](https://github.com/MrDogeBro/sphinx_rtd_dark_mode/blob/main/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)

This Sphinx extension adds a toggleable dark mode to the Read the Docs theme.
A little icon is added in the bottom right hand corner which allows the user
to switch between light or dark mode.

## Installation

**Python 3.4 or higher is required**

To install dark mode for the RTD theme, enter the following command into your
terminal or command prompt.

```bash
# Linux/OSX
python3 -m pip install sphinx-rtd-dark-mode

# Windows
py -3 -m pip install sphinx-rtd-dark-mode
```

## Using the Extension

To use the extension, you will need to add it to the extensions array in your config
file (`conf.py`).

```py
extensions = ["sphinx_rtd_dark_mode"]
```

You will also need to be using the Sphinx RTD theme for this to work. If your not,
then the extension will change it for you as it only works for that theme.

## Config

Below are the options that are customizeable for the extension. Currently, there
is only one option available to configure.

#### Default Dark Mode

This lets you choose which theme the user sees when they load the docs for the first
time ever. After the first time however, this setting has no effect as the users
preference is stored in local storage within their browser. This option accepts a
boolean for the value. If this option is true (the default option), users will start
in dark mode when first visiting the site. If this option is false, users will start
in light mode when they first visit the site.

```py
# user starts in dark mode
default_dark_mode = True

# user starts in light mode
default_dark_mode = False
```

## License

The Sphinx RTD Dark Mode module for Python is licensed under an [MIT license](https://github.com/MrDogeBro/sphinx_rtd_dark_mode/blob/master/LICENSE).
