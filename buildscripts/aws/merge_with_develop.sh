#!/bin/bash
set -exo pipefail

# File for merging with develop. Doesn't work if the merge has to be done manually.

# Set info to avoid errors
git config --global user.email "nick@bodo.ai"
git config --global user.name "Nick Riasanovsky"
git pull --no-edit origin develop
