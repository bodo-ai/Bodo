#!/bin/bash
set -exo pipefail
cd /home/ubuntu/benchmark_logs
git add .
git commit -m "benchmark result"
git push origin master
