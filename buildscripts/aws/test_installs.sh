#!/bin/bash

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

set -eo pipefail
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# s3fs is required by pandas for S3 IO.
# We lock fsspec at version 0.8 because in 0.9 it
# caused us import errors with s3fs for nightly.
conda install -y -c conda-forge boto3 botocore fsspec=0.8 s3fs
conda install -y -c conda-forge pymysql sqlalchemy
conda install -y -c conda-forge scikit-learn='1.0.*' gcsfs
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge pyspark openjdk
conda install -y flake8
pip install pytest pytest-cov deltalake
