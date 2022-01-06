#!/bin/bash
set -e -u -x

# We don't support Python 3.6 or 3.10 (because of Bodo or a dependency of Bodo)
rm -rf /opt/python/cp36-cp36m
rm -rf /opt/python/cp310-cp310

# Install Python packages required to build Bodo pip package. Install for all Python
# versions that we support
for PYBIN in /opt/python/cp*/bin; do
    "${PYBIN}/pip" install astor twine Cython numpy==1.17.* numba==0.54.1 wheel pandas==1.3.* pyarrow==5.0.0
    PYARROW_PATH=`"${PYBIN}/python" -c "import pyarrow; print('/'.join(pyarrow.__file__.split('/')[:-1]))"`
    ln -s $PYARROW_PATH/libarrow.so.500        $PYARROW_PATH/libarrow.so
    ln -s $PYARROW_PATH/libarrow_python.so.500 $PYARROW_PATH/libarrow_python.so
    ln -s $PYARROW_PATH/libparquet.so.500      $PYARROW_PATH/libparquet.so
done
