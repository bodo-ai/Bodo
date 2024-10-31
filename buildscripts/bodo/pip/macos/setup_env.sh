#!/bin/bash
set -e
PYTHON_VERSION=$1
ENVS_PATH=$2
eval "$(micromamba shell hook --shell bash)"
micromamba activate base
micromamba remove -n build-env --all || true-

#- Trim Python version to major.minor
PYTHON_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)
# Replace python version in main.yml
sed -i '' "s/^  - python=.*/  - python=${PYTHON_VERSION}/" "$ENVS_PATH/main.yml"
# Remove bodo.ai channel to use conda-forge pyarrow main
sed -i '' "/  - bodo.ai/d" "$ENVS_PATH/main.yml"
# Remove bodo.ai channel to use conda-forge pyarrow dev
sed -i '' "/  - bodo.ai/d" "$ENVS_PATH/dev.yml"

pip install conda-lock
# Create the lock file
conda-lock -f "$ENVS_PATH"/main.yml -f "$ENVS_PATH"/dev.yml
# Create the environment
micromamba install -y -r /Users/runner/micromamba -n build-env --rc-file /Users/runner/work/_temp/setup-micromamba/.condarc --category main --category dev -f conda-lock.yml

# Activate the environment
micromamba activate build-env
