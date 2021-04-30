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
    $CONDA_INSTALL -c conda-forge "gcc_linux-64>=9.0" "gxx_linux-64>=9.0"
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL -c conda-forge clang_osx-64 clangxx_osx-64
else
    echo "Error in compiler install"
fi


# ---- Conda installs for source build ----
if [ "$RUN_NIGHTLY" != "yes" ];
then
   $CONDA_INSTALL -c conda-forge pyarrow=3.0.0
   # We lock fsspec at version 0.8 because in 0.9 it
   # caused us import errors with s3fs for Pandas tests.
   $CONDA_INSTALL fsspec=0.8 -c conda-forge
   $CONDA_INSTALL pandas='1.2.*' -c conda-forge
   $CONDA_INSTALL numba=0.53.0 -c conda-forge
   $CONDA_INSTALL cython -c conda-forge
   $CONDA_INSTALL mpi4py -c conda-forge
   $CONDA_INSTALL scikit-learn gcsfs -c conda-forge
   $CONDA_INSTALL matplotlib -c conda-forge
   $CONDA_INSTALL hdf5=*=*mpich* -c conda-forge
   $CONDA_INSTALL xlrd xlsxwriter openpyxl -c conda-forge
   if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
else
   if [ "$RUNTIME" != "yes" ];
    then
       $CONDA_INSTALL pytorch=1.8 -c pytorch -c conda-forge -c defaults
       $CONDA_INSTALL bokeh=2.3 -c pytorch -c conda-forge -c defaults
       $CONDA_INSTALL torchvision=0.9 -c pytorch -c conda-forge -c defaults
       # Install h5py and hd5f directly because otherwise tensorflow
       # installs a non-mpi version.
       $CONDA_INSTALL tensorflow h5py hdf5=*=*mpich* -c conda-forge
       pip install horovod;
    fi
   pip install credstash
fi
