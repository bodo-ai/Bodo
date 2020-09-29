#!/bin/sh
echo "Removing the compiled libraries"
find . -name "*.so" | xargs rm -f
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
