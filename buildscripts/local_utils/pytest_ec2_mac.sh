#!/bin/bash

set -ex
# Install python3.14
brew install python@3.14 git

# Clone the repo
export GITHUB_PAT=
git clone https://"$GITHUB_PAT"@github.com/bodo-ai/Bodo.git

# Create a virtual environment
python3.14 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
brew install unzip maven hadoop awscli
pip install pytest wheel setuptools setuptools_scm psutil pyspark boto3 scipy s3fs snowflake-connector-python sqlalchemy snowflake-sqlalchemy scikit-learn mmh3 h5py avro adlfs pytest-azurepipelines pymysql openpyxl
(cd Bodo/iceberg && pip install .)
export HADOOP_HOME="/usr/local/Cellar/hadoop/3.3.0/libexec"
echo "export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home" | tee -a $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Setup environment variables
export HADOOP_HOME="/opt/homebrew/Cellar/hadoop/3.4.0/libexec"
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
export HADOOP_OPTIONAL_TOOLS=hadoop-azure
export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native
export CLASSPATH=$($HADOOP_HOME/bin/hdfs classpath --glob)
export BODO_TRACING_DECRYPTION_FILE_PATH=$(echo "Bodo/buildscripts/decompress_traces.py")
# Pytest EC2 Credentials 1Password
export SF_USERNAME=
export SF_PASSWORD=
export SF_ACCOUNT=
export SF_USER2=
export SF_PASSWORD2=
export SF_AZURE_USER=
export SF_AZURE_PASSWORD=
export SF_AZURE_ACCOUNT=
export AZURE_STORAGE_ACCOUNT_NAME=
export AZURE_STORAGE_ACCOUNT_KEY=
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_SESSION_TOKEN=
export TABULAR_CREDENTIAL=
export AGENT_NAME=1
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
if [[ $NRANKS -gt 1 ]]; then
    export BODO_TESTING_PIPELINE_HAS_MULTI_RANK_TEST=true
else
    export BODO_TESTING_PIPELINE_HAS_MULTI_RANK_TEST=false
fi

# Download wheels
# x86_64
aws s3 cp s3://isaac-test-wheels/cibw-wheels-macos-13-0.zip .
unzip cibw-wheels-macos-13-0.zip

# arm64
#aws s3 cp s3://isaac-test-wheels/cibw-wheels-macos-14-1.zip .
#unzip cibw-wheels-macos-14-1.zip

# Install Bodo
find . -name "bodo*312*.whl" -exec pip install {} \;

# Run PR CI
export NRANKS=2
export PYTEST_MARKER="(not weekly) and (not slow) and (not s3) and (not snowflake) and (not iceberg)"
python -m bodo.runtests "BODO_MAC_PR_CI" "$NRANKS" --pyargs bodo -s -v --import-mode=append -m "$PYTEST_MARKER" || true
