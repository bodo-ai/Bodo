#!/bin/bash
set -exo pipefail

# File that pulls the script needed to run Codebuild in docker on a local machine.

# Refer to this blog post: https://aws.amazon.com/blogs/devops/announcing-local-build-support-for-aws-codebuild/
# for information on how to build images locally.
wget https://raw.githubusercontent.com/aws/aws-codebuild-docker-images/master/local_builds/codebuild_build.sh
chmod +x codebuild_build.sh
