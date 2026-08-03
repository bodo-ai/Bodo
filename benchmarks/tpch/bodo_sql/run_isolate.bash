#!/bin/bash

for q in {1..22}; do
    echo Running $q
    mpiexec -n 1 python bodosql_queries.py --folder s3://tpch-data-parquet/SF1000 --queries $q --scale_factor 1000
done
