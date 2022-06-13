#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail

cd bodo_iceberg_connector/iceberg-reader
mvn clean install -q -Dmaven.test.skip=true -f pom.xml -Dmaven.repo.local=$PREFIX/iceberg-connector-mvn/
cp target/iceberg-reader-1.0-SNAPSHOT.jar $PREFIX/lib/iceberg-reader-1.0-SNAPSHOT.jar
cd -

python setup.py build install --single-version-externally-managed --record=record.txt
