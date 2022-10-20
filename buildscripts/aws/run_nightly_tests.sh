#!/bin/bash
set -exo pipefail

# Load the env first because credstash is installed on conda
export PATH=$HOME/miniconda3/bin:$PATH
export BODO_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`

source activate $CONDA_ENV

USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

# ------ Install Bodo -----------
mamba install -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/bodo-binary -c conda-forge bodo=$BODO_VERSION

# ------ Export environment variables for Snowflake tests -----
export SF_USERNAME=`credstash -r us-east-2 get snowflake.bodopartner.ue1.username`
export SF_PASSWORD=`credstash -r us-east-2 get snowflake.bodopartner.ue1.password`
export SF_ACCOUNT=`credstash -r us-east-2 get snowflake.bodopartner.ue1.account`

export SF_AZURE_USERNAME=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.username`
export SF_AZURE_PASSWORD=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.password`
export SF_AZURE_ACCOUNT=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.account`

# Setup Hadoop (and Arrow) environment variables
export HADOOP_HOME=/opt/hadoop-3.3.2
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS='-Djava.library.path=$HADOOP_HOME/lib'
export HADOOP_OPTIONAL_TOOLS=hadoop-azure
export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

# ------ Environment Variables for iceberg Tests ------
export NESSIE_AUTH_TOKEN=`credstash -r us-east-2 get nessie_auth_token`

# ------ AWS Role ARN for tests -------
export BODO_E2E_TEST_ROLE_ARN_TO_ASSUME=`credstash -r us-east-2 get bodo.engine.nightly.iam_role`

# --------- Run Tests -----------
cd e2e-tests
pytest -s -v --durations=0 --ignore=deep_learning
