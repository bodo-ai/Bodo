#!/bin/bash
set -exo pipefail

# Used to run unit tests inside AWS codebuild

# Deactivate env in case this was called by another file that
# activated the env
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
    export PATH=$HOME/miniforge3/bin:$PATH
    source activate $CONDA_ENV
    set -x

    # Set testing environment variables
    export AZURE_STORAGE_ACCOUNT_NAME=`credstash -r us-east-2 get azure_iceberg_storage_account`
    export AZURE_STORAGE_ACCOUNT_KEY=`credstash -r us-east-2 get azure_iceberg_access_key`
fi

# if running on one core, collect coverage, otherwise run without
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$NP" = "1" ]; then
    # run the tests
    python bodo/runtests.py "Bodo_Tests" "$NP" -s -v -m "$PYTEST_MARKER" --cov-report= --cov=./ bodo/tests
else
    python bodo/runtests.py "Bodo_Tests" "$NP" -s -v -m "$PYTEST_MARKER" bodo/tests
    # Generate an empty coverage file so you can share a single yml file. This should not impact the result
    touch .coverage
fi
