#!/bin/bash

set -e

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1


#pytest -s -v -W ignore --pyargs bodo
#mpiexec -n 2 pytest -s -v -W ignore --pyargs bodo
#mpiexec -n 3 pytest -s -v -W ignore --pyargs bodo
