#!/bin/sh

# Create Ray cluster
ray up daft-cluster.yaml

# request 3 additional workers, wait for them to be ready
ray submit daft-cluster.yaml ../scripts/scale_cluster.py 512

# run the benchmark and write output to S3 bucket
for i in {1..3}; do
    ray submit daft-cluster.yaml iceberg_benchmark.py
done

# cleanup cluster
ray down daft-cluster.yaml