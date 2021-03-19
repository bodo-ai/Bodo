#!/bin/bash

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

set -eo pipefail
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV


conda install -y -c conda-forge boto3 botocore s3fs
conda install -y -c conda-forge pymysql sqlalchemy
conda install -y -c conda-forge scikit-learn gcsfs
conda install -y flake8
pip install pytest pytest-cov deltalake
