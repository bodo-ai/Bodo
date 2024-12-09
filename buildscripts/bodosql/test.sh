#!/bin/bash
set -eo pipefail
set +x

export PATH=$HOME/miniforge3/bin:$PATH
source activate $CONDA_ENV
set -x

# Move to BodoSQL directory
cd BodoSQL

# needed for AWS tests
export BODOSQL_PY4J_GATEWAY_PORT="auto"

# Unlike in run_unitests.sh, we don't need to collect coverage for sonar. As we
# only collect coverage for Bodo.

# unittests
python bodosql/runtests.py "BodoSQL_Tests" 1 -s -v -p no:faulthandler -m "$PYTEST_MARKER" bodosql/tests/

# restore cwd to original directory
cd ..
