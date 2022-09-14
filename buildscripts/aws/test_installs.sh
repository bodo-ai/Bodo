#!/bin/bash
set -exo pipefail

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# s3fs is required by pandas for S3 IO.
mamba install -y -c conda-forge boto3 botocore fsspec>=2021.09 s3fs
mamba install -y -c conda-forge pymysql sqlalchemy
mamba install -y -c conda-forge scikit-learn='1.0.*' gcsfs
mamba install -y -c conda-forge matplotlib
mamba install -y -c conda-forge pyspark=3.2 'openjdk=11'
mamba install -y flake8
mamba install -y -c conda-forge snowflake-sqlalchemy snowflake-connector-python
mamba install -y -c conda-forge mmh3=3.0  # Needed for Iceberg testing
# snowflake connector might upgrade pyarrow, so we revert it back
mamba install -y -c conda-forge pyarrow=8.0.0
# cx_oracle: Oracle Database
# psycopg2: PostgreSQL
if [ "$RUN_NIGHTLY" != "yes" ]; then
    mamba install -y -c conda-forge cx_oracle psycopg2
fi
mamba install -y -c conda-forge pytest pytest-cov
python -m pip install deltalake

conda clean -a -y
