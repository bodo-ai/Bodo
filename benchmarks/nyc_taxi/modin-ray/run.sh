#!/bin/bash

set -eo pipefail

# start cluster
ray up modin-cluster.yaml -y

ray submit modin-cluster.yaml scale_cluster.py

# run full benchmark
ray submit modin-cluster.yaml nyc_taxi_precipitation.py 

# run two more times
ray submit modin-cluster.yaml nyc_taxi_precipitation.py 
ray submit modin-cluster.yaml nyc_taxi_precipitation.py 

# finally, tear down the cluster
ray down modin-cluster.yaml -y

