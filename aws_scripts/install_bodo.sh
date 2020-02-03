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
conda create -n Bodo python=3.7
cd /home/ubuntu
eval "$(conda shell.bash hook)"
conda activate Bodo
conda install -y bodo h5py scipy hdf5=*=*mpich* -c file://home/ubuntu/bodo-inc/ -c bodo.ai -c conda-forge
