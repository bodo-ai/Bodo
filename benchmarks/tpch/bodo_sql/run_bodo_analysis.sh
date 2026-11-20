#!/bin/bash

folder=$1
scale_factor=$2

profile_dir=/bodofs/Users/scott/profiles

for query in {1..22}; do
    echo "Running query $query"
    BODO_TRACING_LEVEL=1 \
    BODO_TRACING_OUTPUT_DIR="$profile_dir/profile_q$query" \
    mpiexec -n 1 python bodosql_queries.py \
    --folder $folder \
    --scale_factor $scale_factor \
    --queries $query \
    --log_timings profile_timings.csv \
    --explain_first

    # Aggregate each profile generated for this query.
    for run_dir in "$profile_dir"/profile_q"$query"/run_*; do
        echo "Aggregating profile: $run_dir"
        mpiexec -n 1 python -m bodo.utils.aggregate_query_profiles "$run_dir" > /dev/null
    done
done