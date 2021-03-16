#!/bin/bash
set -exo pipefail

# Load the env first because credstash is installed on conda
export PATH=$HOME/miniconda3/bin:$PATH
export BODO_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`
source activate $CONDA_ENV

USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

# ------ Install Bodo -----------
conda install -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/bodo-binary -c conda-forge bodo=$BODO_VERSION

# ------ Run Tests -----------
git clone https://github.com/Bodo-inc/engine-e2e-tests.git $ENGINE_E2E_TESTS
cd $ENGINE_E2E_TESTS
pytest -s -v --durations=0
