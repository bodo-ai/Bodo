#!/bin/bash
set -exo pipefail

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1
export BODO_NUM_WORKERS=3

if [[ "$gpu" == "true" && "$is_platform" == "true" ]]; then
    echo "Skipping tests for GPU Platform build since CUDA-Aware MPICH is not available."
elif [[ "$gpu" == "true" ]]; then
    export BODO_GPU_DISABLE_CPU_FALLBACK=1
    python -c "import bodo.pandas as pd; print(pd.DataFrame({'a': [1, 2], 'b': [3, 4]})['a'])"
elif [[ "$(uname)" == "Darwin" ]]; then
    # OpenMPI requires mpiexec on macOS CI.
    mpiexec -n 1 python -u examples/Misc/misc_pi.py
else
    python -u examples/Misc/misc_pi.py
fi
