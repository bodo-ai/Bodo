#!/bin/bash
set -exo pipefail

IS_RELEASE=`git tag --points-at HEAD`
# if it's not a release, publish to the dev channel
if [[ -z $IS_RELEASE ]]; then
    artifactory_channel="bodo.ai-dev"
else
    artifactory_channel="bodo.ai"
fi
echo $artifactory_channel
