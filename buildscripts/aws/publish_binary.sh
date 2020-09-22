#!/bin/bash
set -exo pipefail

echo "********* Zipping Binary **********"
cd $CODEBUILD_SRC_DIR
ls bodo-inc/linux-64
zip -r bodo-linux.zip bodo-inc

echo "********** Publishing to Artifactory **********"
pip install credstash
USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

curl -u${USERNAME}:${TOKEN} -T bodo-linux.zip "https://bodo.jfrog.io/artifactory/bodo-binary/bodo-linux-nightly.zip"
