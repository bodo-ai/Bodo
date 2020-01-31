#!/bin/bash
export GIT_DESCRIBE_TAG=$(git describe --tag)
GIT_DESCRIBE_TAG=${GIT_DESCRIBE_TAG//-/.}
export GIT_DESCRIBE_NUMBER=1
