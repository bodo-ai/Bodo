#!/bin/bash

set -exo pipefail

source activate $CONDA_ENV

if [ "$RUN_COVERAGE" == "yes" ]; then
    coverage combine
    coveralls -v
fi
