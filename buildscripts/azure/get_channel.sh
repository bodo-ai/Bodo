#!/bin/bash
set -eo pipefail

PLATFORM_DEV_RELEASE=${1:-false}
IS_RELEASE=`git tag --points-at HEAD`
if [[ $CHECK_LICENSE_EXPIRED  == 1 ]] && [[ $OBFUSCATE == 1 ]]; then
    # if it's not a release, publish to the dev channel
    if [[ -z $IS_RELEASE ]]; then
        artifactory_channel="bodo.ai-dev"
    else
        artifactory_channel="bodo.ai"
    fi
elif [[ $CHECK_LICENSE_PLATFORM == 1 ]] && [[ $OBFUSCATE != 1 ]]; then
        artifactory_channel="bodo.ai-platform-dev"
elif [[ $CHECK_LICENSE_PLATFORM == 1 ]] && [[ $OBFUSCATE == 1 ]]; then
    # if release or platform-dev-release, then to bodo.ai-platform, else to bodo.ai-platform-dev
    if [[ ! -z "$IS_RELEASE" ]] || [[ "$PLATFORM_DEV_RELEASE" == "true" ]]; then
        artifactory_channel="bodo.ai-platform"
    else
        artifactory_channel="bodo.ai-platform-dev"
    fi
else
    artifactory_channel="bodo-binary"
fi
echo $artifactory_channel
