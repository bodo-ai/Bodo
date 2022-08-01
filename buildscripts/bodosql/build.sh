#!/bin/bash
set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
fi
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# bodo install
python setup.py develop --no-ccache

# NOTE: we need to cd into the directory before building,
# as the run leaves behind a .egg-info in the workigndirectory,
# and if we have multiple of these in the same directory,
# we can run into conda issues.
cd iceberg
# bodo iceberg install
python setup.py develop
cd ..


# bodosql install
cd BodoSQL
python setup.py develop
cd ..
