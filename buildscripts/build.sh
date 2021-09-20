#!/bin/bash

set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV
export BODO_VERSION=` python -c "import versioneer; print(versioneer.get_version())"`

# # install Numba in a directory to avoid import conflict
# mkdir req_install
# pushd req_install
# git clone https://github.com/IntelLabs/numba
# pushd numba
# git checkout hpat_req
# python setup.py install
# popd
# popd

# pushd parquet_reader
# mkdir build
# pushd build
# cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
#     -DCMAKE_INSTALL_LIBDIR=$CONDA_PREFIX/lib -DPQ_PREFIX=$CONDA_PREFIX ..
# make VERBOSE=1
# make install
# popd
# popd

# build Bodo
python setup.py develop --no-ccache
# TODO: fix regular install
# python setup.py build install
