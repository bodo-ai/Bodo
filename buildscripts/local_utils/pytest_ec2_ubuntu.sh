#!/bin/bash
set -ex
# Needs root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root"
    exit 1
fi

# Run on ubuntu 24
# Install python3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 -y
sudo apt install python3.11-venv -y

# Clone the repo
export GITHUB_PAT=
git clone https://"$GITHUB_PAT"@github.com/bodo-ai/Bodo.git

# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
sudo apt install unzip maven openjdk@11 -y
pip install pytest wheel setuptools setuptools_scm psutil pyspark boto3 scipy s3fs snowflake-connector-python sqlalchemy snowflake-sqlalchemy scikit-learn mmh3 h5py avro adlfs pytest-azurepipelines cx_oracle
(cd Bodo/iceberg && pip install -v --no-deps --no-build-isolation .)
wget -q -O - "https://adlsresources.blob.core.windows.net/adlsresources/hadoop-3.3.2.tar.gz" | sudo tar -xzf - -C /opt
echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" | sudo tee -a /opt/hadoop-3.3.2/etc/hadoop/hadoop-env.sh
wget https://download.oracle.com/otn_software/linux/instantclient/215000/instantclient-basic-linux.x64-21.5.0.0.0dbru.zip
sudo $(which unzip) instantclient-basic-linux.x64-21.5.0.0.0dbru.zip -d /usr/local/lib
sudo wget -P /usr/local/bin https://dl.min.io/server/minio/release/linux-amd64/minio
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Setup environment variables
export LD_LIBRARY_PATH=/usr/local/lib/instantclient_21_5:$LD_LIBRARY_PATH
export HADOOP_HOME=/opt/hadoop-3.3.2
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
export HADOOP_OPTIONAL_TOOLS=hadoop-azure
export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native
export CLASSPATH=$($HADOOP_HOME/bin/hdfs classpath --glob)
export BODO_TRACING_DECRYPTION_FILE_PATH=$(echo "Bodo/buildscripts/decompress_traces.py")
export AGENT_NAME=1
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
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Configure what tests to run
export NRANKS=2
export MAX_TEST_PART=30
export START_TEST_PART=1
export END_TEST_PART=10
export PYTEST_MARKER="bodo_${TEST_PART}of${MAX_TEST_PART} and (not weekly) and (not hdfs) and (not iceberg)"
if [[ $NRANKS -gt 1 ]]; then
    export BODO_TESTING_PIPELINE_HAS_MULTI_RANK_TEST=true
else
    export BODO_TESTING_PIPELINE_HAS_MULTI_RANK_TEST=false
fi

# Download wheels
aws s3 cp s3://isaac-test-wheels/cibw-wheels-ubuntu-latest-2.zip .
unzip cibw-wheels-ubuntu-latest-2.zip

# Install Bodo
find . -name "bodo*311*.whl" -exec pip install {} \;

# Run test part 1 through 10
for TEST_PART in $(seq $START_TEST_PART $END_TEST_PART); do
    # Run azure nightly ci
    export PYTEST_MARKER="bodo_${TEST_PART}of${MAX_TEST_PART} and (not weekly) and (not iceberg)"
    python -m bodo.runtests "BODO_${NRANKS}P_${TEST_PART}_OF_${MAX_TEST_PART}" "$NRANKS" --pyargs bodo -s -v --import-mode=append -m "$PYTEST_MARKER"
done
