#!/bin/bash
set -exo pipefail

# Used to run the benchmark.sh file of performance benchmark

cd $BENCHMARK_DEST && ./benchmark.sh
