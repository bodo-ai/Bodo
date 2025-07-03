#!/bin/sh

BUCKET_NAME=$1

if [[ -z "$BUCKET_NAME" ]]; then
  echo "Usage: $0 <bucket-name>"
  exit 1
fi

# Create Ray cluster
ray up daft-cluster.yaml

# request 3 additional workers, wait for them to be ready
ray submit daft-cluster.yaml ../../scripts/scale_cluster.py 256

# run the benchmark and write output to S3 bucket
for i in {1..3}; do
    ray submit daft-cluster.yaml nyc_taxi_precipitation.py --s3_bucket $BUCKET_NAME
done

# cleanup cluster
ray down daft-cluster.yaml
