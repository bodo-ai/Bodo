#!/bin/bash
set -exo pipefail

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


# ---- Create Conda Env ----
CONDA_INSTALL="conda install -q -y"
source deactivate
conda remove --all -q -y -n $CONDA_ENV

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60

if [ "$RUN_NIGHTLY" != "yes" ];
then
    conda create -n $CONDA_ENV -q -y -c conda-forge python=3.8 numpy scipy boost-cpp=1.74.0 cmake h5py mpich mpi
else
    conda create -n $CONDA_ENV -q -y -c conda-forge python=3.8
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
   $CONDA_INSTALL -c conda-forge pyarrow=1.0.1
   $CONDA_INSTALL pandas='1.1.*' -c conda-forge
   $CONDA_INSTALL numba=0.51.2 -c conda-forge
   $CONDA_INSTALL mpi4py -c conda-forge
   $CONDA_INSTALL scikit-learn -c conda-forge
   $CONDA_INSTALL -c pytorch -c conda-forge -c defaults bokeh pytorch=1.5 torchvision=0.6
   pip install horovod[pytorch]
   $CONDA_INSTALL hdf5=*=*mpich* -c conda-forge
   $CONDA_INSTALL xlrd xlsxwriter -c conda-forge
   if [ "$RUN_COVERAGE" == "yes" ]; then $CONDA_INSTALL coveralls; fi
fi

