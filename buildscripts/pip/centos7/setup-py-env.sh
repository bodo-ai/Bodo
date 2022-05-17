#!/bin/bash
set -e -u -x

# We don't support Python 3.6, 3.7 (because of Bodo or a dependency of Bodo)
rm -rf /opt/python/cp36-cp36m
rm -rf /opt/python/cp37-cp37m

# Install Python packages required to build Bodo pip package. Install for all Python
# versions that we support
for PYBIN in /opt/python/cp*/bin; do
    # For Python 3.10, the earliest numpy binaries available are 1.21, and source
    # packages prior to 1.21 do not build.
    if "${PYBIN}/python" -c 'import sys; sys.exit(sys.version_info[:2] >= (3, 10))'; then
        # exit code 0 when version < 3.10
        "${PYBIN}/pip" install astor twine Cython numpy==1.18.* wheel pyarrow==7.0.0
    else
        # exit code 1 when version >= 3.10
        "${PYBIN}/pip" install astor twine Cython numpy==1.21.* wheel pyarrow==7.0.0
    fi
    PYARROW_PATH=`"${PYBIN}/python" -c "import pyarrow; print('/'.join(pyarrow.__file__.split('/')[:-1]))"`
    ln -s $PYARROW_PATH/libarrow.so.700        $PYARROW_PATH/libarrow.so
    ln -s $PYARROW_PATH/libarrow_python.so.700 $PYARROW_PATH/libarrow_python.so
    ln -s $PYARROW_PATH/libparquet.so.700      $PYARROW_PATH/libparquet.so
done
