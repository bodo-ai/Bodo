#!/bin/bash
set -e

# Evaluate the shell.
eval "$(micromamba shell hook --shell bash)"

# Activate the environment
micromamba activate build-env

# Install the Bodo package.
pip install bodo --find-links $(find . -name "cibw-wheels-*" -print) --no-index --no-deps

cd BodoSQL

pip wheel --no-deps --no-build-isolation -v .
