#!/bin/bash
set -xeo pipefail


# Only download miniforge if we're not at runtime.
if [ "$RUNTIME" != "yes" ];
then
  # Install Miniconda
  # Reference:
  # https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh
  unamestr=`uname`
  uname_mach_str=`uname -m`
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
  conda create -n $CONDA_ENV -q -y -c conda-forge python=$PYTHON_VERSION mamba
else 
  if [[ "$uname_mach_str" == 'arm64' ]] || [[ "$uname_mach_str" == 'aarch64' ]]; then
    export MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
    if [[ "$unamestr" == 'Linux' ]]; then
      export MINIFORGE_FILE="Miniforge3-Linux-aarch64.sh"
    elif [[ "$unamestr" == 'Darwin' ]]; then
      export MINIFORGE_FILE="Miniforge3-MacOSX-arm64.sh"
    else
      echo Error
    fi
    curl -L -O "${MINIFORGE_URL}/${MINIFORGE_FILE}"
    bash $MINIFORGE_FILE -b -p $HOME/miniforge3
    export PATH=$HOME/miniforge3/bin:$PATH
  else
    export PATH=$HOME/miniconda3/bin:$PATH
  fi
fi


# ---- Create Conda Env ----
MAMBA_INSTALL="conda install -q -y"

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
fi
source activate $CONDA_ENV

# ---- Install packages from Conda ----

# Needed for BodoSQL
$MAMBA_INSTALL -c conda-forge pytest pytest-cov maven py4j openjdk=11 credstash pyspark=3.2
