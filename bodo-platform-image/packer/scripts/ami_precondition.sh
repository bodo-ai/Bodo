#!/bin/bash

dnf update -y
# Install kernel headers for efa, requires restart
dnf install -y kernel-devel-matched
# Bug in rocky 9.4, updating kernel doesn't trigger rebuilding grub
grub2-mkconfig -o /boot/grub2/grub.cfg

dnf install -y python3-pip
dnf config-manager --set-enabled crb
dnf install -y epel-release
dnf install -y java-11-openjdk-headless.x86_64
dnf install -y unzip

# Install SSM agent
dnf install -y https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/linux_amd64/amazon-ssm-agent.rpm

# Install EFS utils
dnf -y install git rpm-build make rust cargo openssl-devel
git clone https://github.com/aws/efs-utils
(cd efs-utils && make rpm && dnf -y install build/amazon-efs-utils*rpm)
rm -rf efs-utils

pip3 install "ansible==2.9.17" botocore

# install aws cli for downloading Slurm binaries from S3
# not using the preinstalled aws cli v1 to use v2 for both AWS and Azure
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
ln -s /usr/local/bin/aws /usr/bin/aws
reboot
