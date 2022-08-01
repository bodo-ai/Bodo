#!/bin/bash
set -exo pipefail
# Install Minio on AWS Codebuild (no use of Sudo)

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  wget -P /usr/local/bin https://dl.min.io/server/minio/release/linux-amd64/minio
elif [[ "$unamestr" == 'Darwin' ]]; then
  wget -P /usr/local/bin https://dl.min.io/server/minio/release/darwin-amd64/minio
else
  echo Error
fi

chmod +x /usr/local/bin/minio
