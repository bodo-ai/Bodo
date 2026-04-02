#!/bin/bash

set -eo pipefail

curl -L -o mpich-5.0.0.tar.gz https://github.com/pmodels/mpich/releases/download/v5.0.0/mpich-5.0.0.tar.gz
tar -xzf mpich-5.0.0.tar.gz
cd mpich-5.0.0
./configure --prefix=/opt/mpich --disable-doc --disable-dependency-tracking --disable-cxx --disable-fortran --disable-f08 --enable-mpi-abi --with-wrapper-dl-type=none --disable-static --with-hwloc=${CONDA_PREFIX} --with-yaksa=embedded --with-pm=hydra:gforker --with-device=ch4:ucx --with-ucx=${CONDA_PREFIX} --with-cuda=${CONDA_PREFIX}
make -j20
make install