#!/bin/bash
set -eo pipefail
IS_RELEASE=`git tag --points-at HEAD`
if [[ $CHECK_LICENSE_EXPIRED  == 1 ]] && [[ $OBFUSCATE == 1 ]]; then
    # if it's not a release, publish to the dev channel
    if [[ -z $IS_RELEASE ]]; then
        artifactory_channel="bodo.ai-dev"
    else
        artifactory_channel="bodo.ai"
    fi
elif [[ $CHECK_LICENSE_EXPIRED  == 0 ]] && [[ $OBFUSCATE == 1 ]] && [[ ! -z "$IS_RELEASE" ]]; then
        artifactory_channel="bodo.ai-platform"
else
    artifactory_channel="bodo-binary"
fi
echo $artifactory_channel
