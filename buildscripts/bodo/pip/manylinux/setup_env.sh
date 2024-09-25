#!/bin/bash
PYTHON_VERSION=$1
ENVS_PATH=$2
set -e pipefail
echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "ENVS_PATH: $ENVS_PATH"

# Trim Python version to major.minor
PYTHON_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f1,2)

eval "$(micromamba shell hook --shell bash)"
# Replace python version in main.yml
sed  "s/- python=.*/- python=${PYTHON_VERSION}/g" "${ENVS_PATH}/main.yml" > "${ENVS_PATH}/main-mod-version.yml"
# Create the lock file
pip install conda-lock
conda-lock -f "${ENVS_PATH}"/main-mod-version.yml -f "${ENVS_PATH}"/dev.yml -c conda-forge
# Create the environment
micromamba install --category main --category dev -f conda-lock.yml -y --force-reinstall

