#!/bin/bash
set -exo pipefail
# Copied from BodoSQL

# Conda-Build deletes .git, so we need to determine the version beforehand
# and then pass it to setuptools_scm in the build script (setup.py).
# https://conda-forge.org/docs/maintainer/knowledge_base.html#using-setuptools-scm
export SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION"
python setup.py build install --single-version-externally-managed --record=record.txt
