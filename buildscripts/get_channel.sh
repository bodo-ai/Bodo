#!/bin/bash
set -exo pipefail

PLATFORM_DEV_RELEASE=${1:-false}
IS_RELEASE=`git tag --points-at HEAD`
if [[ $IS_BODO_PLATFORM == 1 ]]; then
    # if release or platform-dev-release, then to bodo.ai-platform, else to bodo.ai-platform-dev
    if [[ ! -z "$IS_RELEASE" ]] || [[ "$PLATFORM_DEV_RELEASE" == "true" ]]; then
        artifactory_channel="bodo.ai-platform"
    else
        artifactory_channel="bodo.ai-platform-dev"
    fi
else
    artifactory_channel="bodo.ai"
fi
echo $artifactory_channel
