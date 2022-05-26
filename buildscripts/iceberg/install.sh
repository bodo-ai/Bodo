#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail


# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh
unamestr=`uname`
# For M1, we need to use miniforge to enable cross compilation
if [[ "$TARGET_NAME" == 'macOS_ARM' ]]; then
  export MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
  export MINIFORGE_FILE="Miniforge3-MacOSX-x86_64.sh"
  curl -L -O "${MINIFORGE_URL}/${MINIFORGE_FILE}"
  bash $MINIFORGE_FILE -b -p $HOME/miniforge3
  export PATH=$HOME/miniforge3/bin:$PATH
else
  if [[ "$unamestr" == 'Linux' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  elif [[ "$unamestr" == 'Darwin' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  else
    echo Error
  fi
  chmod +x miniconda.sh
  ./miniconda.sh -b
  export PATH=$HOME/miniconda3/bin:$PATH
fi


# ---- Create Conda Env ----
source deactivate
conda remove --all -q -y -n build_iceberg_connector
conda create -n build_iceberg_connector -q -y -c conda-forge python=$PYTHON_VERSION
source activate build_iceberg_connector

# ---- Install packages from Conda ----
conda install -q -y -c conda-forge maven jpype1 credstash
