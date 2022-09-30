#!/bin/bash
set -exo pipefail

unamestr=`uname`

# Install Miniconda
# Reference:
# https://github.com/numba/numba/blob/master/buildscripts/incremental/install_miniconda.sh
if [ "$RUNTIME" != "yes" ];
then
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
MAMBA_INSTALL="mamba install -q -y"
# Deactivate if another script has already activated the env
source deactivate || true

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60

if [ "$RUNTIME" != "yes" ];
then
  conda create -n $CONDA_ENV -q -y -c conda-forge python=$PYTHON_VERSION mamba
fi
source activate $CONDA_ENV

if [ "$RUNTIME" != "yes" ];
then
  if [ "$RUN_NIGHTLY" != "yes" ];
  then
      $MAMBA_INSTALL -c conda-forge numpy scipy boost-cpp=1.74.0 cmake h5py mpich mpi
  else
      $MAMBA_INSTALL -q -y -c conda-forge cmake make
  fi
fi


# ---- install compilers ----
if [[ "$unamestr" == 'Linux' ]]; then
    $MAMBA_INSTALL -c conda-forge 'gcc_linux-64>=9' 'gxx_linux-64>=9'
elif [[ "$unamestr" == 'Darwin' ]]; then
    $MAMBA_INSTALL -c conda-forge clang_osx-64 clangxx_osx-64
else
    echo "Error in compiler install"
    exit 1
fi


# ---- Conda installs for source build ----
if [ "$RUN_NIGHTLY" != "yes" ];
then
   $MAMBA_INSTALL -c conda-forge boost-cpp=1.74.0 cmake h5py mpich mpi
   $MAMBA_INSTALL 'hdf5=1.12.*=*mpich*' -c conda-forge
   $MAMBA_INSTALL -c conda-forge pyarrow=8.0.0
   $MAMBA_INSTALL fsspec>=2021.09 -c conda-forge
   $MAMBA_INSTALL pandas=${BODO_PD_VERSION:-'1.4.*'} -c conda-forge
   $MAMBA_INSTALL numba=0.55.2 -c conda-forge
   $MAMBA_INSTALL cython -c conda-forge
   $MAMBA_INSTALL mpi4py -c conda-forge
   $MAMBA_INSTALL scikit-learn='1.1.*' 'gcsfs>=2022.1' -c conda-forge
   $MAMBA_INSTALL matplotlib='3.5.1' -c conda-forge
   $MAMBA_INSTALL pyspark=3.2 'openjdk=11' -c conda-forge
   $MAMBA_INSTALL xlrd xlsxwriter openpyxl -c conda-forge
   if [ "$RUN_COVERAGE" == "yes" ]; then $MAMBA_INSTALL coveralls; fi
else
   if [ "$RUNTIME" != "yes" ];
    then
       conda clean -a -y
       #$MAMBA_INSTALL pytorch=1.9 pyarrow=8.0.0 -c pytorch -c conda-forge -c defaults
       $MAMBA_INSTALL pyarrow=8.0.0 -c conda-forge
       conda clean -a -y
       #$MAMBA_INSTALL bokeh=2.3 -c pytorch -c conda-forge -c defaults
       #conda clean -a -y
       #$MAMBA_INSTALL torchvision=0.10 -c pytorch -c conda-forge -c defaults
       #conda clean -a -y
       # Install h5py and hd5f directly because otherwise tensorflow
       # installs a non-mpi version.
       #$MAMBA_INSTALL tensorflow h5py hdf5=$HDF5_VERSION -c conda-forge
       # If building the docker image always install 1.12 for hdf5
       $MAMBA_INSTALL h5py 'hdf5=1.12.*=*mpich*' -c conda-forge
       conda clean -a -y
       #pip install horovod;
    fi
   pip install credstash
fi

conda clean -a -y
