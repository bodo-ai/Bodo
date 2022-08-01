#!/bin/bash
set -exo pipefail
# Copied from BodoSQL

python setup.py build install --single-version-externally-managed --record=record.txt
