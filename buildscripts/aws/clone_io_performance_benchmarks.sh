#!/bin/bash

# Clones the IO Performance benchmark repo found here:
# https://github.com/Bodo-inc/performance_benchmarks

set -eo pipefail
git clone https://github.com/Bodo-inc/performance_benchmarks $BENCHMARK_DEST
