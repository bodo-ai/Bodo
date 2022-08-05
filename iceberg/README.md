# Bodo Iceberg Connector

See https://bodo.atlassian.net/wiki/spaces/B/pages/1018593350/Iceberg+Dev+Setup#Installations

To build:

    # Needs to be done only once
    conda install -c conda-forge 'openjdk>=9.0,<12' py4j maven pyspark=3.2
    # Java package built automatically
    python setup.py develop
