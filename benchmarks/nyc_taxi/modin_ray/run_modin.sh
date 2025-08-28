#!/bin/bash

set -eo pipefail

# start cluster
ray up modin-cluster.yaml -y

# scale cluster up to 256 vCPUs
# ray submit modin-cluster.yaml ../../scripts/scale_cluster.py 256

# run full benchmark
# WARNING! This might take up to 3 hours.
ray submit modin-cluster.yaml nyc_taxi_precipitation.py
# ray submit modin-cluster.yaml nyc_taxi_precipitation.py
# ray submit modin-cluster.yaml nyc_taxi_precipitation.py

# finally, tear down the cluster
# ray down modin-cluster.yaml -y

