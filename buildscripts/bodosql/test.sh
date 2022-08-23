#!/bin/bash
set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
fi
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# Move to BodoSQL directory
cd BodoSQL

# needed for AWS tests
export BODOSQL_PY4J_GATEWAY_PORT="auto"

# Unlike in run_unitests.sh, we don't need to collect coverage for sonar. As we
# only collect coverage for Bodo.

# unittests
python bodosql/runtests.py 1 -s -v -p no:faulthandler -m "$PYTEST_MARKER" bodosql/tests/

# restore cwd to original directory
cd ..
