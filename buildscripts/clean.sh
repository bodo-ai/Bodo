#!/bin/sh
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
echo "Removing bodo/io/csv_json_reader.cpp"
find . -name "csv_json_reader.cpp" | xargs rm -f
echo "Removing bodo/io/_hdfs.cpp"
find . -name "_hdfs.cpp" | xargs rm -f
echo "Removing mpi4py"
rm -rf bodo/mpi4py
