#!/bin/bash

set -eo pipefail

# start cluster
ray up modin-cluster.yaml -y

# run a smaller job to make sure all of the workers are up and running
ray submit modin-cluster.yaml nyc_taxi_preciptation.py -d small

# run the full benchmark
ray submit modin-cluster.yaml nyc_taxi_preciptation.py -d large

# finally, tear down the cluster
ray down modin-cluster.yaml -y

