#!/bin/bash
# Copied from BodoSQL with updates to use micromamba
set -xeo pipefail


# Install Micromamba. Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh
wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"  # Limited time Use


# ---- Setup Base Env ----
# Unlike miniforge / mambaforge, micromamba does not come with anything (including Python)
# preinstalled, so the base environment is completely empty.
micromamba activate  # this activates the base environment
micromamba install -q -y python=$PYTHON_VERSION maven jpype1 credstash -c conda-forge
