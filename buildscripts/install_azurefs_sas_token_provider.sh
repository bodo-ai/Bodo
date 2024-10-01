#!/bin/bash
set -exo pipefail

if [[ "$CI_SOURCE" == "AWS" ]]; then
    export PATH=$HOME/miniforge3/bin:$PATH

    # ---- Activate Conda Env ----
    # Deactivate if another script has already activated the env
    set +x
    source deactivate || true
    source activate $CONDA_ENV
    set -x
fi

# Setup Hadoop
wget -q -O - "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-3.3.2/hadoop-3.3.2.tar.gz" | tar -xzf - -C /tmp

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

cd azurefs-sas-token-provider
# TODO: Install pip on Docker image to properly build
python setup.py develop
cd ..
