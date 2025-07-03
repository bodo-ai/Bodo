#!/bin/sh
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
echo "Removing bodo/io/csv_json_reader.cpp"
find . -name "csv_json_reader.cpp" | xargs rm -f
echo "Removing vendored mpi4py"
rm -rf bodo/mpi4py/_vendored_mpi4py
echo "Removing DuckDB build"
rm -rf bodo/pandas/vendor/duckdb/build
