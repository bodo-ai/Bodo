#!/bin/bash

source activate $CONDA_ENV

# generate test data for test_io
python bodo/tests/gen_test_data.py

if [ "$RUN_COVERAGE" == "yes" ]; then
    export PYTHONPATH=.
    coverage erase
    coverage run --source=./bodo --omit ./bodo/ml/*,./bodo/xenon_ext.py,./bodo/ros.py,./bodo/cv_ext.py,./bodo/tests/gen_test_data.py -m unittest
else
    mpiexec -n $NUM_PES python -u -m unittest -v
fi
