#!/bin/bash
set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
IS_RELEASE=${3:-}

echo "********** Publishing to Artifactory **********"
USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

for package in `ls $HOME/miniconda3/envs/bodo_build/conda-bld/${OS_DIR}/bodo*.tar.bz2`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
done

# reindex conda
ADMIN_USERNAME=`credstash -r us-east-2 get artifactory.admin.username`
ADMIN_TOKEN=`credstash -r us-east-2 get artifactory.admin.token`
curl -X POST https://$ADMIN_USERNAME:$ADMIN_TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex

# wait for reindex to complete
set +e
for package in `ls $HOME/miniconda3/envs/bodo_build/conda-bld/${OS_DIR}/bodo*.tar.bz2`; do
    package_name=`basename $package`
    # package name looks like this: bodo-<version>-<build>.tar.bz2
    # for example: bodo-2022.05.2+10.gb5b9a120.dirty-py39h3fd9d12_10.tar.bz2
    export BODO_VERSION=`python -c "print('$package_name'.split('-')[1])"`
    export BODO_BUILD=`python -c "print('$package_name'.split('-')[2].split('.')[0])"`
    exit_status=1
    while [[ $exit_status != 0 ]]
    do
        sleep 30
        conda search bodo="${BODO_VERSION}" -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/${CHANNEL_NAME}/${OS_DIR} | grep "$BODO_BUILD"
        exit_status=$?
    done
done
