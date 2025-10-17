#!/bin/bash
set -exo pipefail

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
BODO_VERSION=${3:-}

echo "********** Publishing to Artifactory **********"
label="main"
PACKAGE_DIR=$HOME/conda-bld/$OS_DIR

for package in `ls $PACKAGE_DIR/bodo*.conda`; do
    package_name=`basename $package`
    if [[ "$CHANNEL_NAME" != "DO_NOT_PUBLISH" ]]; then
        curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
    fi
    if [[ ! -z "$label" ]]; then
        anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
    fi
done

if [[ "$CHANNEL_NAME" != "DO_NOT_PUBLISH" ]]; then
    curl -X POST https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex
fi

# Block on checking if the reindex has failed.
set +e
exit_status=1
while [[ $exit_status != 0 ]]
do
    sleep 30
    conda search bodo="${BODO_VERSION}" -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${CHANNEL_NAME}/${OS_DIR}
    exit_status=$?
done
