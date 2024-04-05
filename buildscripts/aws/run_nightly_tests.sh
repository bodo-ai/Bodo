#!/bin/bash
set -exo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    # Load the env first because credstash is installed on conda
    export PATH=$HOME/mambaforge/bin:$PATH

    source deactivate || true

    set +x
    source activate $CONDA_ENV
    set -x

    # All stored as Github Secrets for Github Actions. This is for Codebuild
    # ------ Export environment variables for Snowflake tests -----
    export SF_USERNAME=`credstash -r us-east-2 get snowflake.bodopartner.ue1.username`
    export SF_PASSWORD=`credstash -r us-east-2 get snowflake.bodopartner.ue1.password`
    export SF_ACCOUNT=`credstash -r us-east-2 get snowflake.bodopartner.ue1.account`

    export SF_AZURE_USERNAME=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.username`
    export SF_AZURE_PASSWORD=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.password`
    export SF_AZURE_ACCOUNT=`credstash -r us-east-2 get snowflake.kl02615.east-us-2.azure.account`

    # ------ Environment Variables for iceberg Tests ------
    export NESSIE_AUTH_TOKEN=`credstash -r us-east-2 get nessie_auth_token`
fi

# ------ Setup Hadoop (and Arrow) environment variables ------
export HADOOP_HOME=/tmp/hadoop-3.3.2
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

# ------ Clean Maven and Spark Ivy Cache ------
rm -rf $HOME/.ivy2/cache $HOME/.ivy2/jars $HOME/.m2/repository

# --------- Run Tests -----------
cd e2e-tests
pytest -s -v --durations=0 --ignore=deep_learning
cd ..
pytest -s -v --durations=0 bodo/tests/test_javascript*

