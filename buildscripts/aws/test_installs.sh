#!/bin/bash

# Installations needed to run unittests and sonarqube. All placed in 1 file for the AWS Codebuild install step.

set -eo pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV
conda install -y -c conda-forge boto3 botocore "s3fs>=0.4.2"
conda install -y -c conda-forge pymysql sqlalchemy
conda install -y flake8
pip install pytest

if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$NP" = "1" ]; then
    # install coverage and credstash
    pip install pytest-cov credstash

    # sonar download and setup
    wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.4.0.2170-linux.zip
    unzip sonar-scanner-cli-4.4.0.2170-linux.zip

    echo "sonar.host.url=http://ec2-35-175-128-216.compute-1.amazonaws.com:9000/" >> sonar-scanner-4.4.0.2170-linux/conf/sonar-scanner.properties
fi
