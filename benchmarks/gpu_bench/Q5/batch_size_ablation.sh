#!/bin/bash

# args:
#  <root path prefix> (e.g. s3://my-bucket/tpch-data/)
#  Should contain directories SF100, SF1000, etc with the TPCH data in parquet format
root_dir=$1

batch_sizes=(12_000_000 24_000_000 48_000_000 96_000_000)
scale_factors=(100 1000)
n_workers=(1 2)

for n in "${n_workers[@]}"; do
    for sf in "${scale_factors[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            python run_bodo.py \
                --root $root_dir/SF$sf \
                --batch_size $batch_size \
                --n_iters 3 \
                --n_workers $n \
                --batch_size $batch_size \
                --warmup
        done
    done
done
