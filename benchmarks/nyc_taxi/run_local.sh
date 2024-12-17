#!/bin/bash

# Run all systems. Note: uses sleep to ensure resources are reset before
# running the next system.

python -m nyc_taxi.local_versions -s dask
sleep 4

python -m nyc_taxi.local_versions -s bodo
sleep 4

python -m nyc_taxi.local_versions -s modin
sleep 4

python -m nyc_taxi.local_versions -s spark

