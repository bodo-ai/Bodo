#!/bin/bash

set -exo pipefail

export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
export CONDA_BUILD_SYSROOT=$(xcrun --show-sdk-path)

echo "Using sysroot: $CONDA_BUILD_SYSROOT"
