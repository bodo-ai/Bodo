#!/bin/bash -xe

# Used to run unit tests inside AWS codebuild

set -eo pipefail
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV
flake8 bodo


# if running on one core, collect coverage, otherwise run without
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$NP" = "1" ]; then
    # run the tests
    python bodo/runtests.py "$NP" -s -v -m "$PYTEST_MARKER" --cov-report= --cov=./ bodo/tests
else
    python bodo/runtests.py "$NP" -s -v -m "$PYTEST_MARKER" bodo/tests
    # Generate an empty coverage file so you can share a single yml file. This should not impact the result
    touch .coverage
fi
