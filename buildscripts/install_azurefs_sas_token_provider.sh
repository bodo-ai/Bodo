#!/bin/bash
set -exo pipefail

# Setup Hadoop
if [[ ! -f hadoop.tar.gz ]]; then
    wget -O hadoop.tar.gz "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-3.3.2/hadoop-3.3.2.tar.gz"
fi
tar -xzf hadoop.tar.gz -C /tmp

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
pip install .
cd ..
