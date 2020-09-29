#!/bin/bash

# Used to run the benchmark.sh file of performance benchmark

set -eo pipefail

cd $BENCHMARK_DEST && ./benchmark.sh
