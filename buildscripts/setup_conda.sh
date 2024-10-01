#!/bin/bash
set -exo pipefail

unamestr=`uname`

# Install Miniforge
if [[ "$unamestr" == 'Linux' ]]; then
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
elif [[ "$unamestr" == 'Darwin' ]]; then
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -O miniforge.sh
else
  echo Error
fi
chmod +x miniforge.sh
./miniforge.sh -b
export PATH=$HOME/miniforge3/bin:$PATH


# ---- Create Conda Env ----
# Deactivate if another script has already activated the env
set +x
source deactivate || true
set -x

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60
# Conda / Mamba will attempt to upgrade packages listed in the
# "aggressive_update_packages" at every CLI call.
# To upgrade these packages, it will also upgrade packages that depend on
# them. OpenSSL is usually one of the defaults, and many packages Bodo
# uses depends on OpenSSL, thus breaking our dependency locking.
# Therefore, we need to remove all entries from this setting
conda config --add aggressive_update_packages nodefaults

mamba install -y -c conda-forge conda-lock
conda-lock install --dev --mamba -n $CONDA_ENV buildscripts/envs/conda-lock.yml

# This docker image is LARGE, so we remove unnecessary files
# after the environment is created.
# TODO: Check how long this takes at PR CI runtime
conda clean -a -y
