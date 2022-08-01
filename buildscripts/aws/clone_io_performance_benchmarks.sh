#!/bin/bash
set -exo pipefail

# Clones the IO Performance benchmark repo found here:
# https://github.com/Bodo-inc/performance_benchmarks

git clone https://github.com/Bodo-inc/performance_benchmarks $BENCHMARK_DEST
