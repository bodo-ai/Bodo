#!/bin/bash
set -exo pipefail

# Package Setup
eval "$(micromamba shell hook -s posix)"
micromamba activate bodo_build

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
BODO_VERSION=${3:-}

echo "********** Publishing to Artifactory **********"
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`
ANACONDA_TOKEN=`cat $HOME/secret_file | grep anaconda.org.token | cut -f 2 -d' '`

label="main"
PACKAGE_DIR=$HOME/conda-bld/$OS_DIR

for package in `ls $PACKAGE_DIR/bodo*.conda`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
    if [[ ! -z "$label" ]]; then
        anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
    fi
done

curl -X POST https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex

# Block on checking if the reindex has failed.
set +e
exit_status=1
while [[ $exit_status != 0 ]]
do
    sleep 30
    conda search bodo="${BODO_VERSION}" -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${CHANNEL_NAME}/${OS_DIR}
    exit_status=$?
done
