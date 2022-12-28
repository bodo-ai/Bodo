#!/bin/bash
set -eo pipefail

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# Print Command before Executing
# Source Operations Prints Too Much Output
set -x

# s3fs is required by pandas for S3 IO.
mamba install -y -c conda-forge boto3 botocore fsspec>=2021.09 s3fs
mamba install -y -c conda-forge pymysql sqlalchemy
mamba install -y -c conda-forge scikit-learn='1.1.*' gcsfs
mamba install -y -c conda-forge matplotlib
mamba install -y -c conda-forge pyspark=3.2 'openjdk=11'
mamba install -y flake8
mamba install -y -c conda-forge snowflake-connector-python snowflake-sqlalchemy 'openjdk=11'
mamba install -y -c conda-forge mmh3=3.0 'openjdk=11' # Needed for Iceberg testing
# snowflake connector might upgrade pyarrow, so we revert it back
mamba install -y -c conda-forge pyarrow=9.0.0 'openjdk=11'
# cx_oracle: Oracle Database
# psycopg2: PostgreSQL
if [ "$RUN_NIGHTLY" != "yes" ]; then
    mamba install -y -c conda-forge cx_oracle psycopg2 'openjdk=11'
fi
mamba install -y -c conda-forge pytest pytest-cov pytest-timeout 'openjdk=11'
# From snowflake to cx_oracle or psycopg2 all 
# downgrade hdf5 and install it without mpi
# So we revert it back.
mamba install -y -c conda-forge 'hdf5=1.12.*=*mpich*' 'openjdk=11'
python -m pip install deltalake
python -m pip install awscli

conda clean -a -y
