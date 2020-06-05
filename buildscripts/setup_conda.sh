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

conda create -n $CONDA_ENV -q -y -c conda-forge python numpy scipy boost-cpp cmake h5py mpich mpi
source activate $CONDA_ENV

# install compilers
if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL -c conda-forge clang_osx-64=9.0.1 clangxx_osx-64=9.0.1
else
    echo "Error in compiler install"
fi

$CONDA_INSTALL -c conda-forge pyarrow=0.17.1
$CONDA_INSTALL pandas>=1.0.0 -c conda-forge
$CONDA_INSTALL numba=0.49.1 -c conda-forge
$CONDA_INSTALL mpi4py -c conda-forge
$CONDA_INSTALL -c conda-forge hdf5=*=*mpich*
$CONDA_INSTALL -c conda-forge xlrd xlsxwriter

if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
