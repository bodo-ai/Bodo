#!/bin/bash

set -exo pipefail

./get_miniconda.sh
CONDA_INSTALL="conda install -q -y"

source deactivate

conda remove --all -q -y -n $CONDA_ENV

conda create -n $CONDA_ENV -q -y -c conda-forge python numpy scipy boost-cpp=1.74.0 cmake h5py mpich mpi
source activate $CONDA_ENV

# install compilers
if [[ "$unamestr" == 'Linux' ]]; then
    $CONDA_INSTALL gcc_linux-64 gxx_linux-64 gfortran_linux-64
elif [[ "$unamestr" == 'Darwin' ]]; then
    $CONDA_INSTALL -c conda-forge clang_osx-64 clangxx_osx-64
else
    echo "Error in compiler install"
fi

$CONDA_INSTALL -c conda-forge pyarrow=1.0.1
$CONDA_INSTALL pandas='1.1.*' -c conda-forge
$CONDA_INSTALL numba=0.51.2 -c conda-forge
$CONDA_INSTALL mpi4py -c conda-forge
$CONDA_INSTALL scikit-learn -c conda-forge
$CONDA_INSTALL hdf5=*=*mpich* -c conda-forge
$CONDA_INSTALL xlrd xlsxwriter -c conda-forge

if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
