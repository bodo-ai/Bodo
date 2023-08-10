#!/bin/bash

yum install -y python3-pip
yum install epel-release -y
yum install -y java-11-openjdk-headless.x86_64

pip3 install --upgrade pip
pip3 install "ansible==2.9.17"
pip3 install ansible[azure]

# install aws cli for downloading Slurm binaries from S3
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
