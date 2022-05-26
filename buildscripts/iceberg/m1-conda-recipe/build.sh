#!/bin/bash
# Copied from BodoSQL
set -xeo pipefail

cd bodo_iceberg_connector/iceberg-reader
mvn clean install -Dmaven.test.skip=true -f pom.xml -Dmaven.repo.local=$PREFIX/iceberg-connector-mvn/
cp target/iceberg-reader-1.0-SNAPSHOT.jar $PREFIX/lib/iceberg-reader-1.0-SNAPSHOT.jar
cd -

MACOSX_DEPLOYMENT_TARGET=11.0 \
python setup.py build install --single-version-externally-managed --record=record.txt
