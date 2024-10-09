#!/bin/bash
dnf update -y
dnf config-manager --set-enabled crb
dnf install -y epel-release
dnf install -y python3-pip unzip cifs-utils java-11-openjdk-headless.x86_64

# A dependency of ansible[azure] (azure-keyvault) requires pip 23 or lower because of 
# a weird issue in the metadata. Got the error:
# Requested azure-keyvault==1.0.0a1 from https://files.pythonhosted.org/packages/03/f3/fe18493d4ce781368f23d05701a8203344fdc15dbf9cfee4450652776d1a/azure_keyvault-1.0.0a1-py2.py3-none-any.whl (from ansible[azure]==2.9.17) has invalid metadata: Expected matching RIGHT_PARENTHESIS for LEFT_PARENTHESIS, after version specifier azure-arm.azure-arm-build:     msrest (>=0.4.17azure-common~=1.1.5)
python -m pip install pip~=23.0
python -m pip install "ansible[azure]==2.9.17" 

# install aws cli for downloading Slurm binaries from S3
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
