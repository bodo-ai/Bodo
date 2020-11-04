#!/bin/bash

# Installations needed to run unittests. All placed in 1 file for the AWS Codebuild install step.

set -eo pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

conda install -y -c conda-forge boto3 botocore "s3fs>=0.4.2"
conda install -y -c conda-forge pymysql sqlalchemy
conda install -y -c conda-forge scikit-learn
conda install -c pytorch -c conda-forge -c defaults bokeh pytorch=1.5 torchvision=0.6
pip install horovod[pytorch]
conda install -y flake8
pip install pytest pytest-cov

