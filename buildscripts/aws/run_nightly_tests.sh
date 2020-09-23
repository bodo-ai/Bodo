#!/bin/bash
set -exo pipefail

USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

# ------ Conda Config -----------
export PATH=$HOME/miniconda3/bin:$PATH
conda remove --all -q -y -n $CONDA_ENV
conda create -n $CONDA_ENV

# ------ Test Installs -----------
cd $CODEBUILD_SRC_DIR
./buildscripts/aws/test_installs.sh

# ------ Install Bodo -----------
conda install -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/bodo-binary/bodo-2020.09-py38hc547734_19.tar.bz2 -c conda-forge bodo

# ------ Run Tests -----------
git clone https://github.com/Bodo-inc/engine-e2e-tests.git $ENGINE_E2E_TESTS
cd $ENGINE_E2E_TESTS
pytest
