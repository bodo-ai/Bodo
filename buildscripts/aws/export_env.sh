#!/bin/bash
set -eo pipefail
set +x

# Deactivate env in case this was called by another file that activated the env
source deactivate || true
export PATH=$HOME/miniforge3/bin:$PATH
source activate $CONDA_ENV

# Print Command before Executing
# Source Operations Prints Too Much Output
set -x

mamba env export | tee environment.yml
mamba list --explicit > package-list.txt
pip list | tee requirements.txt
