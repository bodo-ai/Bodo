#!/bin/bash -xe

# Used to run unit tests inside AWS codebuild

set -eo pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV
flake8 bodo




# if running on one core, collect coverage and run Sonar, otherwise run without
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$NP" = "1" ]; then

    # get the sonar token used to authenticate against the sonar server from credstash
    TOKEN=`credstash --kms-region us-east-2 get sonar.analysis.token`
    PULL_REQUEST_ID=`echo $CODEBUILD_WEBHOOK_TRIGGER | cut -f2 -d/`

    # run the tests
    python bodo/runtests.py "$NP" -s -v -m "$PYTEST_MARKER"  --cov-report= --cov=./ bodo/tests

    # run the sonar scanner analysis passing in the pullrequest configuration to enable decorators on the PR
    sonar-scanner-4.4.0.2170-linux/bin/sonar-scanner -Dsonar.login=$TOKEN  -Dsonar.pullrequest.key=$PULL_REQUEST_ID -Dsonar.pullrequest.branch=$CODEBUILD_SOURCE_VERSION

else
    python bodo/runtests.py "$NP" -s -v -m "$PYTEST_MARKER" bodo/tests
fi
