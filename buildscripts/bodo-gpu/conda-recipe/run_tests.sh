#!/bin/bash

set -eo pipefail

export BODO_GPU=1
export OMPI_MCA_pml="ucx"
export BODO_NUM_WORKERS=3

pytest -svWignore -m "gpu" bodo/tests/test_end_to_end.py
