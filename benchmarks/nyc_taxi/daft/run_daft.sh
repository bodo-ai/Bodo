#!/bin/sh

ray up daft-cluster.yaml

# request 3 additional workers, wait for them to be ready
ray submit daft-cluster.yaml ../scripts/scale_cluster.py 256

ray submit daft-cluster.yaml nyc_taxi_precipitation.py

ray down daft-cluster.yaml
