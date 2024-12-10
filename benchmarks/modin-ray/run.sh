#!/bin/bash

set -eo pipefail

# start cluster
ray up modin-cluster.yaml -y

# run full benchmark
ray submit modin-cluster.yaml nyc_taxi_precipitation.py 

# finally, tear down the cluster
ray down modin-cluster.yaml -y

