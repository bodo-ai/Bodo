#!/bin/bash
set -exo pipefail

export PATH=$HOME/miniconda3/bin:$PATH


# ---- Create Conda Env ----
CONDA_INSTALL="conda install -q -y"
# Deactivate if another script has already activated the env
source deactivate || true

# Set 5 retries with 1 minute in between to try avoid HTTP errors
conda config --set remote_max_retries 5
conda config --set remote_backoff_factor 60
source activate $CONDA_ENV

$CONDA_INSTALL -c conda-forge openjdk jpype1 maven pyspark=3.2
cd iceberg/bodo_iceberg_connector/iceberg-reader
mvn clean install -Dmaven.test.skip=true -f pom.xml
cd ../..
python setup.py develop
