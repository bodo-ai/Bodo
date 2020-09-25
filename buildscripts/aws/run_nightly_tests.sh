#!/bin/bash
set -exo pipefail

USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# ------ Install Bodo -----------
sub_channel=`cat $CODEBUILD_SRC_DIR/bodo_subchannel`
conda install -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/bodo-binary/$sub_channel -c conda-forge bodo

# ------ Run Tests -----------
git clone https://github.com/Bodo-inc/engine-e2e-tests.git $ENGINE_E2E_TESTS
cd $ENGINE_E2E_TESTS
pytest
