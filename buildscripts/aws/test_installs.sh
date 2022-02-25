#!/bin/bash

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

set -eo pipefail
# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# s3fs is required by pandas for S3 IO.
conda install -y -c conda-forge boto3 botocore fsspec>=2021.09 s3fs
conda install -y -c conda-forge pymysql sqlalchemy
conda install -y -c conda-forge scikit-learn='1.0.*' gcsfs
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge pyspark openjdk
conda install -y flake8
# cx_oracle: Oracle Database
# psycopg2: PostgreSQL
if [ "$RUN_NIGHTLY" != "yes" ]; then
    conda install -y -c conda-forge cx_oracle psycopg2
fi
pip install pytest pytest-cov deltalake
