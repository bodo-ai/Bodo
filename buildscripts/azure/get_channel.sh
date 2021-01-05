#!/bin/bash
set -eo pipefail

NUMBA_DEV=${1:-false}
IS_RELEASE=`git tag --points-at HEAD`
if [[ $NUMBA_DEV == "true" ]]; then
    # If numba-dev pipeline, publish to bodo-numba-dev channel
    artifactory_channel="bodo-numba-dev"
elif [[ $CHECK_LICENSE_EXPIRED  == 1 ]] && [[ $OBFUSCATE == 1 ]]; then
    # if it's not a release, publish to the dev channel
    if [[ -z $IS_RELEASE ]]; then
        artifactory_channel="bodo.ai-dev"
    else
        artifactory_channel="bodo.ai"
    fi
elif [[ $CHECK_LICENSE_PLATFORM_AWS == 1 ]] && [[ $OBFUSCATE != 1 ]]; then
        artifactory_channel="bodo.ai-platform-dev"
elif [[ $CHECK_LICENSE_PLATFORM_AWS == 1 ]] && [[ $OBFUSCATE == 1 ]]; then
    # if release, then to bodo.ai-platform, else to bodo.ai-platform-dev
    if [[ ! -z "$IS_RELEASE" ]]; then
        artifactory_channel="bodo.ai-platform"
    else
        artifactory_channel="bodo.ai-platform-dev"
    fi
else
    artifactory_channel="bodo-binary"
fi
echo $artifactory_channel
