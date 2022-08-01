#!/bin/bash
# We can't add -x or -v, as we explcitly use the output to this function
set -eo pipefail
# Script to convert batch specific variable into an S3 prefix
# for use in generating and extracting artifacts across builds
# in a single batch

# EXAMPLE:
# CODEBUILD_INITIATOR=coverage-codebuild-batch-allow/AWSCodeBuild-8319b329-6b3e-402b-974f-832852eec482/Bodo-PR-Testing:17ca8ee1-09ce-43ee-8b87
# Output=17ca8ee1-09ce-43ee-8b87

IFS='/'
read -a strarr <<< "$CODEBUILD_INITIATOR"
batch_name="${strarr[${#strarr[*]} - 1]}"
IFS='-'
read -a strarr <<< "$batch_name"
# Cut off the initial AWSCodeBuild section
batch_arr=${strarr[@]:1}
batch_name=$(echo ${batch_arr[0]} | tr " " -)
IFS=':'
echo $batch_name
