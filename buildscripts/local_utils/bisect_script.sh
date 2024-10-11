#!/bin/bash

# Helper script to be used with 'git bisect run' when identifying commits that break tests.
# Must be ran in the root of the Bodo repository.
# Usage: git bisect run ./bisect_script.sh <pytest command>
# Example: git bisect run ./bisect_script.sh pytest -k Bodosql/bodosql/tests/test_window/test_rank_window_fns.py::test_row_number_filter_multicolumn
# WARNING: cannot distinguish between different test failures,
# or other one off failures (random segfaults, maven build failures, etc)

set -exo pipefail

#Build everything
pip install --no-deps --no-build-isolation -ve .; cd iceberg; pip install -v --no-deps --no-build-isolation .; cd ../BodoSQL; pip install -v --no-deps --no-build-isolation .; cd ..;

#Run the testing command supplied by the user
$@
