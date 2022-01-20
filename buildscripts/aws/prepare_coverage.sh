#!/bin/bash -xe

# Used to download and merge files that will be used by sonar
set -eo pipefail

BATCH_PREFIX=coverage-$(buildscripts/aws/get_batch_prefix.sh)

# Download a set of files from AWS
# TODO(Nick): Replace the bucketname with an env var
python3 buildscripts/aws/download_s3paths_with_prefix.py bodo-pr-testing-artifacts $BATCH_PREFIX

# Update the coverage configuration to resolve absolute paths
python3 buildscripts/aws/update_coverage_config.py

# Merge the coverage files and produce the output for Sonar
coverage combine $BATCH_PREFIX*/.coverage
# Output the final report (may be unnecessary)
coverage xml -i --omit bodo/runtests.py
