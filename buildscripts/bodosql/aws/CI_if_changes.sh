#!/bin/bash
set -exo pipefail

# Used to run unit tests inside AWS codebuild

set -eo pipefail
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

python3 buildscripts/bodosql/aws/CI_if_changes.py $@
