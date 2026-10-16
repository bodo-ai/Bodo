#!/bin/bash
set -exo pipefail

export NUMBA_DEVELOPER_MODE=1
export NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
export PYTHONFAULTHANDLER=1
export BODO_NUM_WORKERS=3
# Don't let UCX try to use RDMA on Github actions/azure runners
# sometimes it fails to initialize because of this
export UCX_TLS="tcp,self,sm"

python -u test.py
