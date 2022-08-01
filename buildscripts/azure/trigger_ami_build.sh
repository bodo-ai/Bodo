#!/bin/bash
set -exo pipefail

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
PLATFORM_DEV_RELEASE=${3:-false}
BODO_VERSION=${4:-}


bot_token=`cat $HOME/secret_file | grep bot.herman.github.token | cut -f 2 -d' '`

echo "Getting the tag of the latest release of the AMI repo"
release_tag=$(curl -H "Authorization: token $bot_token" --silent "https://api.github.com/repos/Bodo-inc/bodo-ami/releases/latest" | jq -r .tag_name)
echo "release_tag: $release_tag"

echo "BODO_VERSION: ${BODO_VERSION}"
if [[ -z "$BODO_VERSION" ]]; then
    echo "BODO_VERSION is empty. Exiting..."
    exit 1
fi

if [[ "$CHANNEL_NAME" == "bodo.ai-platform" ]] && [[ "$OS_DIR" == "linux-64"* ]]; then

    if [[ "$PLATFORM_DEV_RELEASE" == "true" ]]; then
        echo "Is a Dev Only Platform Release..."
        ONLY_DEV="true"
    else
        echo "Is a Regular Platform Release..."
        ONLY_DEV="false"
    fi
    echo "ONLY_DEV: $ONLY_DEV"

    echo "Triggering the AMI-CI on the release_tag"

    curl \
    -X POST \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $bot_token" \
    https://api.github.com/repos/Bodo-inc/bodo-ami/actions/workflows/build_publish_images.yml/dispatches \
    -d '{"ref":"refs/tags/'$release_tag'", "inputs":{"bodoVersion":"'$BODO_VERSION'","onlyDev":"'$ONLY_DEV'"}}'

else
    echo "Skipping..."
fi
