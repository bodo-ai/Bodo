#!/bin/bash

# Used to run unit tests inside AWS codebuild

set -eo pipefail
set +x
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true

# TODO: Do we need to activate here?
export PATH=$HOME/miniforge3/bin:$PATH
# Run with || true in case this is called by a CI build that doesn't
# use conda (Sonar)
source activate $CONDA_ENV || true

set -x
python3 buildscripts/aws/CI_if_changes.py $@
