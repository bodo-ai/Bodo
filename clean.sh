#!/bin/sh
echo "Removing the compiled libraries"
find . -name "*.so" | xargs rm -f
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
echo "Removing bodo/io/pyfs.h and bodo/io/pyfs.cpp"
find . -name "pyfs.cpp" | xargs rm -f
find . -name "pyfs.h" | xargs rm -f
find . -name "_hdfs.cpp" | xargs rm -f
echo "Removing bodo/utils/tracing.c"
find . -name "tracing.c" | xargs rm -f
