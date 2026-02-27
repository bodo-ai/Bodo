#!/bin/bash

# run polars
rm -f timings.csv

# run polars with dask engine (LocalCUDACluster)
for sf in 10 100; do
    for n in 4 2 1; do
        if [ $sf -eq 100 ]; then
            # Dask engine is really slow
            python run_polars.py --engine dask --n_workers $n \
                    --root ../data/tpch/SF$sf
        else
            for i in {1..3}; do
                python run_polars.py --engine dask --n_workers $n \
                    --root ../data/tpch/SF$sf
            done
        fi
    done

    # run polars with engine="gpu"
    for i in {1..3}; do
        python run_polars.py --root ../data/tpch/SF$sf
    done
done

python aggregate_timings.py