#!/bin/bash -xe

# File for merging with master. Doesn't work if the merge has to be done manually.

set -eo pipefail

# Set info to avoid errors
git config --global user.email "nick@bodo.ai"
git config --global user.name "Nick Riasanovsky"
git pull --no-edit origin master