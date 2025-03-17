#!/bin/sh

# Create a random bucket to store the output
BUCKET_NAME=nyc-taxi-benchmark-daft-$(uuidgen | tr -d - | tr '[:upper:]' '[:lower:]' )

aws s3api create-bucket \
    --bucket $BUCKET_NAME \
    --region us-east-2 \
    --create-bucket-configuration LocationConstraint=us-east-2

# Create Ray cluster
ray up daft-cluster.yaml

# request 3 additional workers, wait for them to be ready
ray submit daft-cluster.yaml ../scripts/scale_cluster.py 256

# run the benchmark and write output to S3 bucket
for i in {1..3}; do
    ray submit daft-cluster.yaml nyc_taxi_precipitation.py --s3_bucket $BUCKET_NAME
done

# cleanup cluster
ray down daft-cluster.yaml