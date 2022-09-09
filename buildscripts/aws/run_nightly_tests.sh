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

# ------ Export environment variables for Snowflake tests -----
export SF_USERNAME=`credstash -r us-east-2 get snowflake.bodopartner.ue1.username`
export SF_PASSWORD=`credstash -r us-east-2 get snowflake.bodopartner.ue1.password`
export SF_ACCOUNT=`credstash -r us-east-2 get snowflake.bodopartner.ue1.account`

# ------ Environment Variables for iceberg Tests ------
export NESSIE_AUTH_TOKEN=`credstash -r us-east-2 get nessie_auth_token`

# --------- Run Tests -----------
cd e2e-tests
pytest -s -v --durations=0 --ignore=deep_learning
