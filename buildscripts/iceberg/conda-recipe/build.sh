#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail

python setup.py build install --single-version-externally-managed --record=record.txt
