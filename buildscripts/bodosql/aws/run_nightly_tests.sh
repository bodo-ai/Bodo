#!/bin/bash
set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
fi
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# ------ Export environment variables for Snowflake tests -----
export SF_USERNAME=`credstash -r us-east-2 get snowflake.bodopartner.ue1.username`
export SF_PASSWORD=`credstash -r us-east-2 get snowflake.bodopartner.ue1.password`
export SF_ACCOUNT=`credstash -r us-east-2 get snowflake.bodopartner.ue1.account`

# ------ Run Tests -----------
cd ./BodoSQL-Customer-Examples/
source deactivate || true
source activate $CONDA_ENV
python -m pytest -s -v -p no:faulthandler --durations=0
