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
# Replace bodo.ai::pyarrow with conda-forge::pyarrow
sed  "s_bodo.ai/linux-64::__g" "$ENVS_PATH/main-mod-version.yml" > "${ENVS_PATH}/main-mod-version-2.yml"
# Remove bodo.ai channel to use conda-forge pyarrow in main
sed  "/  - bodo.ai/d" "$ENVS_PATH/main-mod-version-2.yml" > "${ENVS_PATH}/main-mod-version-3.yml"
# Remove bodo.ai channel to use conda-forge pyarrow in dev
sed  "/  - bodo.ai/d" "$ENVS_PATH/dev.yml" > "${ENVS_PATH}/dev-mod-version.yml"
# Create the lock file
pip install conda-lock
conda-lock -f "${ENVS_PATH}"/main-mod-version-3.yml -f "${ENVS_PATH}"/dev-mod-version.yml -c conda-forge
# Create the environment
micromamba install --category main --category dev -f conda-lock.yml -y --force-reinstall

