#!/bin/bash
set -eo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    source deactivate || true
fi
export PATH=$HOME/mambaforge/bin:$PATH

set +x
source activate $CONDA_ENV
set -x

# Enable Sccache to use and save C/C++ cache to S3
export SCCACHE_BUCKET=engine-codebuild-cache
export SCCACHE_REGION=us-east-2
export SCCACHE_S3_USE_SSL=true
export SCCACHE_S3_SERVER_SIDE_ENCRYPTION=true

# TODO: Couple of things to Improve
#   - Use a different role for PR CI or merge roles
#   - Debug why IMDSv2 in sccache is not working for AWS containers
ASSUME_ROLE_CREDENTIALS=`aws sts assume-role --role-arn arn:aws:iam::427443013497:role/BodoEngineNightlyRole --role-session-name BodoEnginePRSession`
export AWS_ACCESS_KEY_ID=`jq -r .Credentials.AccessKeyId <<< "$ASSUME_ROLE_CREDENTIALS"`
export AWS_SECRET_ACCESS_KEY=`jq -r .Credentials.SecretAccessKey <<< "$ASSUME_ROLE_CREDENTIALS"`
export AWS_SESSION_TOKEN=`jq -r .Credentials.SessionToken <<< "$ASSUME_ROLE_CREDENTIALS"`


# bodo install
python setup.py develop --no-ccache

# NOTE: we need to cd into the directory before building,
# as the run leaves behind a .egg-info in the working directory,
# and if we have multiple of these in the same directory,
# we can run into conda issues.
cd iceberg
# bodo iceberg install
python setup.py develop
cd ..


# bodosql install
cd BodoSQL
python setup.py develop
cd ..
