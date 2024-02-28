#!/bin/bash
set -exo pipefail

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
BODO_VERSION=${3:-}

echo "********** Publishing to Artifactory **********"
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`
ANACONDA_TOKEN=`cat $HOME/secret_file | grep anaconda.org.token | cut -f 2 -d' '`

# ARM builds use miniforge to cross compile instead of conda
export PATH=$HOME/mambaforge/bin:$PATH
set +x
source activate bodo_build
set -x

IS_MAJOR_RELEASE=0
if [[ -n "$IS_RELEASE" ]]; then
    IS_MAJOR_RELEASE=`python -c "import os; print(int(len(os.environ[\"IS_RELEASE\"].split(\".\")) < 3))"`
fi
label=""
if [[ "$OBFUSCATE" == 1 ]] && [[ "$IS_MAJOR_RELEASE" == 1 ]]\
    && [[ "$CHANNEL_NAME" == "bodo.ai" ]]; then
    # If we have a major release upload with our main anaconda label
    label="main"
elif [[ "$OBFUSCATE" == 1 ]] && [[ "$CHANNEL_NAME" == "bodo.ai" ]]\
    && [[ -n "$IS_RELEASE" ]]; then
    # If we have a minor release upload with our dev anaconda label
    label="dev"
fi

PACKAGE_DIR=$HOME/conda-bld/$OS_DIR

for package in `ls $PACKAGE_DIR/bodo*.tar.bz2 $PACKAGE_DIR/bodo*.conda`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
    if [[ ! -z "$label" ]]; then
        anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
    fi
done

curl -X POST https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex

# Block on checking if the reindex has failed.
set +e
if [[ $OS_DIR != "osx-arm64" ]]; then
    exit_status=1
    while [[ $exit_status != 0 ]]
    do
        sleep 30
        conda search bodo="${BODO_VERSION}" -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${CHANNEL_NAME}/${OS_DIR}
        exit_status=$?
    done
fi
