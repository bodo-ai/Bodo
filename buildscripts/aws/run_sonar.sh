#!/bin/bash -xe

# Used to upload files to sonarqube

set -eo pipefail
  
# get the sonar token used to authenticate against the sonar server from credstash
TOKEN=`credstash --kms-region us-east-2 get sonar.analysis.token`
PULL_REQUEST_ID=`echo $CODEBUILD_WEBHOOK_TRIGGER | cut -f2 -d/`


# run the sonar scanner analysis passing in the pullrequest configuration to enable decorators on the PR
sonar-scanner-4.4.0.2170-linux/bin/sonar-scanner -Dsonar.login=$TOKEN  -Dsonar.pullrequest.key=$PULL_REQUEST_ID -Dsonar.pullrequest.branch=$CODEBUILD_SOURCE_VERSION -Dsonar.scm.revision=$CODEBUILD_RESOLVED_SOURCE_VERSION
