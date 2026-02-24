#!/bin/bash
set -exo pipefail

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1
export BODO_NUM_WORKERS=3

if [[ "$(uname)" == "Darwin" ]]; then
    # OpenMPI requires mpiexec on macOS CI.
    mpiexec -n 1 python -u examples/Misc/misc_pi.py
else
    python -u examples/Misc/misc_pi.py
fi
