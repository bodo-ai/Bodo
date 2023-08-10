#!/bin/bash

script=$NODE_ROLE
script+="_test.py"

# Bodo-platform-utils requires this to be set
export BODO_PLATFORM_CLOUD_PROVIDER=AWS

# https://github.com/openucx/ucx/issues/4742#issuecomment-584059909
export UCX_TLS=ud,sm,self

sudo mv /tmp/tests $REMOTE_DIR
sudo cp /home/bodo/.bashrc /tmp/.bashrc_tmp

source /tmp/.bashrc_tmp

pip install pytest pytest-testinfra paramiko
python -m pytest -v $REMOTE_DIR/tests/$script
