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

conda create -n $CONDA_ENV -q -y -c conda-forge python=3.7 numpy scipy pandas=0.25.3 boost-cpp cmake h5py mpich mpi
source activate $CONDA_ENV

# install compilers
if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL -c conda-forge clang_osx-64 clangxx_osx-64 gfortran_osx-64
else
    echo "Error in compiler install"
fi

$CONDA_INSTALL -c bodo.ai -c conda-forge pyarrow=0.15.1 arrow-cpp=0.15.1=*transfer_s3*
$CONDA_INSTALL numba=0.47.* -c numba/label/dev
$CONDA_INSTALL -c conda-forge hdf5=*=*mpich*

if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
