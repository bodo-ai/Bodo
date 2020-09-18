#!/bin/bash
set -eo pipefail
# Script to convert batch and build specific variables into an artifact
# name that can be placed in an S3 bucket for sonarqube

IFS=':'
read -a strarr <<< "$CODEBUILD_BUILD_ID"
build_id="${strarr[${#strarr[*]} - 1]}"
echo coverage-$(buildscripts/aws/get_batch_prefix.sh)-$build_id