#!/bin/bash
cd /tmp
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
ls
pwd
echo $PATH
export PATH=/root/miniconda3/bin:$PATH
echo $PATH
conda create -n Bodo python
cd /home/ubuntu
eval "$(conda shell.bash hook)"
conda activate Bodo
HDF5_VERSION=`python -c "import sys; print('1.12.*=*mpich*') if sys.version_info.minor == 10 else print('1.10.*=*mpich*')"`
conda install -y bodo h5py scipy hdf5=$HDF5_VERSION -c file://home/ubuntu/bodo-inc/ -c bodo.ai -c conda-forge
