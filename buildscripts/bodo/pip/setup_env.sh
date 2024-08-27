#!/bin/bash
set -e
PYTHON_VERSION=$1
ENVS_PATH=$2

# Delete the environment if it exists
#micromamba remove -n build-env --all -y || true

# Replace python version in main.yml
sed -i '' "s/^ - python=.*/ - python=${PYTHON_VERSION}/" "$ENVS_PATH/main.yml"
# Create the lock file
pipx run conda-lock -f "$ENVS_PATH"/main.yml -f "$ENVS_PATH"/dev.yml
# Create the environment
micromamba create -y -r /Users/runner/micromamba -n build-env --rc-file /Users/runner/work/_temp/setup-micromamba/.condarc --category main --category dev -f conda-lock.yml 

# Activate the environment and install mpich and mpi4py from mpi4py channel bdist wheels
eval "$(micromamba shell hook --shell bash)"
micromamba activate build-env
micromamba remove --force -y mpich mpi4py
micromamba run -n build-env python -m pip install -i https://pypi.anaconda.org/mpi4py/simple mpich==3.4.3 mpi4py==3.1.5
