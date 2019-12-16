#!/bin/bash

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh

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
export PATH=$HOME/miniconda3/bin:$PATH

CONDA_INSTALL="conda install -q -y"

source deactivate

conda remove --all -q -y -n $CONDA_ENV

conda create -n $CONDA_ENV -q -y numpy scipy pandas boost cmake h5py pyarrow mpich mpi
source activate $CONDA_ENV

# install compilers
if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL gcc_linux-64 gxx_linux-64 gfortran_linux-64
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL clang_osx-64 clangxx_osx-64 gfortran_osx-64
else
    echo "Error in compiler install"
fi

# TODO: remove dev channel when 0.47 is fully released
$CONDA_INSTALL numba=0.47.* -c numba/label/dev
$CONDA_INSTALL -c defaults -c conda-forge hdf5=*=*mpich*

if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
