#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1

python -m bodo.tests.gen_test_data

python -u -m bodo.runtests -v
mpiexec -n 2 python -u -m bodo.runtests -v
mpiexec -n 3 python -u -m bodo.runtests -v
