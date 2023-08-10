#!/bin/bash

yum install -y amazon-efs-utils python3-pip
amazon-linux-extras install -y epel
amazon-linux-extras install -y java-openjdk11

pip3 install "ansible==2.9.17"

# install aws cli for downloading Slurm binaries from S3
# not using the preinstalled aws cli v1 to use v2 for both AWS and Azure
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
