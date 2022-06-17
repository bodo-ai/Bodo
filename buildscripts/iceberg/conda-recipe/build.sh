#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail

cd bodo_iceberg_connector/iceberg-java
mvn clean install --batch-mode -Dmaven.test.skip=true -f pom.xml
cd -

python setup.py build install --single-version-externally-managed --record=record.txt
