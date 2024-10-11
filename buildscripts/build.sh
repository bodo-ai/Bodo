#!/bin/bash
set -eo pipefail

# Deactivate env in case this was called by another file that
# activated the env. This only happens on AWS and causes errors
# on Azure with MacOS
if [[ "$CI_SOURCE" == "AWS" ]]; then
    export PATH=$HOME/miniforge3/bin:$PATH
    source deactivate || true

    set +x
    source activate $CONDA_ENV
    set -x

    # TODO: Couple of things to Improve
    #   - Use a different role for PR CI or merge roles
    #   - Debug why IMDSv2 in sccache is not working for AWS containers
    ASSUME_ROLE_CREDENTIALS=`aws sts assume-role --role-arn arn:aws:iam::427443013497:role/BodoEngineNightlyRole --role-session-name BodoEnginePRSession`
    export AWS_ACCESS_KEY_ID=`jq -r .Credentials.AccessKeyId <<< "$ASSUME_ROLE_CREDENTIALS"`
    export AWS_SECRET_ACCESS_KEY=`jq -r .Credentials.SecretAccessKey <<< "$ASSUME_ROLE_CREDENTIALS"`
    export AWS_SESSION_TOKEN=`jq -r .Credentials.SessionToken <<< "$ASSUME_ROLE_CREDENTIALS"`
fi

# Enable Sccache to use and save C/C++ cache to S3
# TODO: Should we move directly into CMake?
export SCCACHE_BUCKET=engine-codebuild-cache
export SCCACHE_REGION=us-east-2
export SCCACHE_S3_USE_SSL=true
export SCCACHE_S3_SERVER_SIDE_ENCRYPTION=true
# We should just use sccache directly on CI
export DISABLE_CCACHE=1
# Build Bodo with our fork of Arrow
export USE_BODO_ARROW_FORK=1

# Bodo Install
pip install --no-deps --no-build-isolation -Ccmake.verbose=true -ve .

# NOTE: we need to cd into the directory before building,
# as the run leaves behind a .egg-info in the working directory,
# and if we have multiple of these in the same directory,
# we can run into conda issues.

# Bodo-Iceberg Install
cd iceberg
pip install --no-deps --no-build-isolation -ve .
cd ..

# BodoSQL Install
cd BodoSQL
pip install --no-deps --no-build-isolation -ve .
cd ..
