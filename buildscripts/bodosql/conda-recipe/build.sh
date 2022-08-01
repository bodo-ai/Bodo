#!/bin/bash
set -exo pipefail

python setup.py build install --single-version-externally-managed --record=record.txt
