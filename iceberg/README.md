# Bodo Iceberg Connector

See https://bodo.atlassian.net/wiki/spaces/B/pages/1018593350/Iceberg+Dev+Setup#Installations

To build:

    # Needs to be done only once
    conda install -c conda-forge openjdk jpype1 maven pyspark=3.2

    cd bodoicebergconnector/iceberg-reader
    mvn clean install -Dmaven.test.skip=true -f pom.xml
    cd ../..
    python setup.py develop
