#!/bin/sh

ray up polars-cluster.yaml

for i in {1..3}; do
    ray submit polars-cluster.yaml nyc_taxi_precipitation.py
done

ray down polars-cluster.yaml
