#!/bin/bash
set -exo pipefail

CHANNEL_NAME=${1:-bodo-binary}
OS_DIR=${2:-linux-64}
IS_RELEASE=${3:-}

echo "********** Publishing to Artifactory **********"
pip install credstash
USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

for package in `ls $HOME/miniconda3/envs/bodo_build/conda-bld/${OS_DIR}/bodo*.tar.bz2`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${CHANNEL_NAME}/${OS_DIR}/$package_name"
    echo "$package_name" > $CODEBUILD_SRC_DIR/bodo_subchannel
done

# reindex conda
ADMIN_USERNAME=`credstash -r us-east-2 get artifactory.admin.username`
ADMIN_TOKEN=`credstash -r us-east-2 get artifactory.admin.token`
curl -X POST https://$ADMIN_USERNAME:$ADMIN_TOKEN@bodo.jfrog.io/artifactory/api/conda/$CHANNEL_NAME/reindex
