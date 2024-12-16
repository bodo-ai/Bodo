#!/bin/bash
set -exo pipefail

# Conda-Build deletes .git, so we need to determine the version beforehand
# and then pass it to setuptools_scm in the build script (setup.py).
# https://conda-forge.org/docs/maintainer/knowledge_base.html#using-setuptools-scm
export SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION"
python -m pip wheel \
    --wheel-dir=/tmp/wheelhouse \
    --no-deps --no-build-isolation -vv .

python -m pip install --no-index --find-links=/tmp/wheelhouse bodosql
