#!/bin/bash

# Install Minio

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  sudo wget -P /usr/local/bin https://dl.min.io/server/minio/release/linux-amd64/minio
elif [[ "$unamestr" == 'Darwin' ]]; then
  sudo wget -P /usr/local/bin https://dl.min.io/server/minio/release/darwin-amd64/minio
else
  echo Error
fi

sudo chmod +x /usr/local/bin/minio
