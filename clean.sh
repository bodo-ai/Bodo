#!/bin/sh
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
echo "Removing bodo/io/pyfs.cpp"
find . -name "pyfs.cpp" | xargs rm -f
echo "Removing bodo/io/_hdfs.cpp"
find . -name "_hdfs.cpp" | xargs rm -f
echo "Removing mpi4py"
find . -name "bodo/mpi4py" | xargs rm -rf
