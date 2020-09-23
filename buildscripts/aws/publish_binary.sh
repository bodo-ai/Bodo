#!/bin/bash
set -exo pipefail

# echo "********* Zipping Binary **********"
# cd $CODEBUILD_SRC_DIR
# ls bodo-inc/linux-64
# zip -r bodo-linux.zip bodo-inc

# echo "********** Publishing to Artifactory **********"
pip install credstash
# USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
# TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

# for package in `ls $HOME/miniconda3/envs/bodo_build/conda-bld/linux-64/bodo*.tar.bz2`; do
#     package_name=`basename $package`
#     curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/bodo-binary/$package_name"
# done
    
