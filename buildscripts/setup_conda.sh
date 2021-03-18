#!/bin/bash
set -exo pipefail

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh
if [ "$RUNTIME" != "yes" ];
then
  unamestr=`uname`
  if [[ "$unamestr" == 'Linux' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  elif [[ "$unamestr" == 'Darwin' ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  else
    echo Error
  fi
    chmod +x miniconda.sh
    ./miniconda.sh -b
fi
export PATH=$HOME/miniconda3/bin:$PATH


# ---- Create Conda Env ----
CONDA_INSTALL="conda install -q -y"
# Deactivate if another script has already activated the env
source deactivate || true

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60
if [ "$RUNTIME" != "yes" ];
then
  if [ "$RUN_NIGHTLY" != "yes" ];
  then
      conda create -n $CONDA_ENV -q -y -c conda-forge python numpy scipy boost-cpp=1.74.0 cmake h5py=2.10 mpich mpi
  else
      conda create -n $CONDA_ENV -q -y -c conda-forge python cmake make
  fi
fi
source activate $CONDA_ENV


# ---- install compilers ----
if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL gcc_linux-64 gxx_linux-64
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL -c conda-forge clang_osx-64 clangxx_osx-64
else
    echo "Error in compiler install"
fi


# ---- Conda installs for source build ----
if [ "$RUN_NIGHTLY" != "yes" ];
then
   $CONDA_INSTALL -c conda-forge pyarrow=3.0.0
   $CONDA_INSTALL pandas='1.2.*' -c conda-forge
   $CONDA_INSTALL numba=0.53.0 -c conda-forge
   $CONDA_INSTALL cython -c conda-forge
   $CONDA_INSTALL mpi4py -c conda-forge
   $CONDA_INSTALL scikit-learn -c conda-forge
   $CONDA_INSTALL hdf5=*=*mpich* -c conda-forge
   $CONDA_INSTALL xlrd xlsxwriter openpyxl -c conda-forge
   if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
else
   $CONDA_INSTALL -c pytorch -c conda-forge -c defaults bokeh pytorch=1.5 torchvision=0.6
   $CONDA_INSTALL -c conda-forge tensorflow
   if [ "$RUNTIME" != "yes" ]; then pip install horovod[pytorch,tensorflow]; fi
   pip install credstash
fi
