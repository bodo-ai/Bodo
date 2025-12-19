#!/bin/sh

# Create Ray cluster
ray up daft-cluster-single-node.yaml

# run the benchmark and write output to S3 bucket
for i in {1..3}; do
    ray submit daft-cluster-single-node.yaml nyc_taxi_precipitation.py --single_node
done

# cleanup cluster
ray down daft-cluster-single-node.yaml
