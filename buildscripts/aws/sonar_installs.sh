#!/bin/bash
set -exo pipefail

# Installations needed for sonar

# Install boto3 so we can read the coverage files from S3
pip3 install boto3
pip3 install botocore

# Install coverage and credstash
pip3 install coverage
pip3 install credstash

# sonar download and setup
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.4.0.2170-linux.zip
unzip sonar-scanner-cli-4.4.0.2170-linux.zip

echo "sonar.host.url=http://ec2-35-175-128-216.compute-1.amazonaws.com:9000/" >> sonar-scanner-4.4.0.2170-linux/conf/sonar-scanner.properties
