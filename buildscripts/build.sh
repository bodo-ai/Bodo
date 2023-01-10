#!/bin/bash

set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/mambaforge/bin:$PATH
source activate $CONDA_ENV
export BODO_VERSION=` python -c "import versioneer; print(versioneer.get_version())"`

# Build Bodo
python setup.py develop --no-ccache
# TODO: fix regular install
# python setup.py build install
