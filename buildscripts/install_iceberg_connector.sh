#!/bin/bash
set -exo pipefail

export PATH=$HOME/mambaforge/bin:$PATH

# Deactivate if another script has already activated the env
set +x
source deactivate || true
source activate $CONDA_ENV
set -x

cd iceberg
# TODO: Install pip on Docker image to properly build
python setup.py develop
cd ..
