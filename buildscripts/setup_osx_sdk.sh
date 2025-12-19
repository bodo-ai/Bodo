#!/bin/bash

set -exo pipefail

export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
export CONDA_BUILD_SYSROOT="/opt/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk"
echo "Downloading ${MACOSX_DEPLOYMENT_TARGET} sdk"
curl -L -O https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk.tar.xz
sudo tar -xf MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk.tar.xz -C /opt
