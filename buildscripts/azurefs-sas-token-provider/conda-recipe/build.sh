#!/bin/bash
set -exo pipefail
# Copied from BodoSQL

python -m pip wheel \
    --wheel-dir=/tmp/wheelhouse \
    --no-deps --no-build-isolation -vv .

python -m pip install --no-index --find-links=/tmp/wheelhouse bodo-azurefs-sas-token-provider
