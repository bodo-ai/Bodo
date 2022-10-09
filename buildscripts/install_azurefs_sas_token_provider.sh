#!/bin/bash
set -exo pipefail

export PATH=$HOME/miniconda3/bin:$PATH


# ---- Create Conda Env ----
CONDA_INSTALL="conda install -q -y"
# Deactivate if another script has already activated the env
source deactivate || true

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60
source activate $CONDA_ENV

# Setup Hadoop
$CONDA_INSTALL -c conda-forge 'openjdk=11' maven
wget -q -O - "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-3.3.2/hadoop-3.3.2.tar.gz" | tar -xzf - -C /opt
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

cd azurefs-sas-token-provider
python setup.py develop
cd ..
