#!/bin/bash
set -exo pipefail

# We don't support Python 3.6, 3.7 (because of Bodo or a dependency of Bodo)
rm -rf /opt/python/cp36-cp36m
rm -rf /opt/python/cp37-cp37m

cd /bodo
# obfuscate
cd obfuscation
/opt/python/cp310-cp310/bin/python do_obfuscation.py
cd ..

# Amend commit to remove dirty from version

# Remove existing tag if it's a release
if [[ ! -z $IS_RELEASE ]]; then
    ## Remove existing tag if it's a release
    git -c user.name='Bot Herman' -c user.email='bot_herman@bodo.ai' tag -d $IS_RELEASE
fi

## Commit the changes after obfuscation
git -c user.name='Bot Herman' -c user.email='bot_herman@bodo.ai' add .
git -c user.name='Bot Herman' -c user.email='bot_herman@bodo.ai' commit -a --amend --no-edit

if [[ ! -z $IS_RELEASE ]]; then
    ## Tag new commit with the old tag, if it's a release
    git -c user.name='Bot Herman' -c user.email='bot_herman@bodo.ai' tag -a $IS_RELEASE -m $IS_RELEASE
fi

HDF5_LD_PATH="/usr/lib64/mpich/lib"

# build Bodo wheels
for PYBIN in /opt/python/cp*/bin; do
    "${PYBIN}/pip" install "mpi4py_mpich==3.1.2"
    PYARROW_PATH=`"${PYBIN}/python" -c "import pyarrow; print('/'.join(pyarrow.__file__.split('/')[:-1]))"`
    HDF5_INCLUDE_PATH="/usr/include/mpich-x86_64"
    CONDA_PREFIX='' CFLAGS='-I/mpich/include -I'$PYARROW_PATH'/include -I'$HDF5_INCLUDE_PATH' -L/mpich/lib -L'$PYARROW_PATH' -L'$HDF5_LD_PATH' -Wl,-rpath,/mpich/lib -Wl,-rpath,'$PYARROW_PATH'' "${PYBIN}/python" setup.py bdist_wheel
done

# We need to add the HDFS_LD_PATH to LD_LIBRARY_PATH before running
# auditwheel so that it can find the .so files.
export LD_LIBRARY_PATH=$HDF5_LD_PATH:$LD_LIBRARY_PATH

# auditwheel repair bundles any external libraries that are not standard to all Linux
# systems into Bodo package, and patches binaries to point to them
for WHL in dist/bodo*whl; do
    auditwheel repair --plat manylinux2014_x86_64 $WHL
done

# Need to undo some of the changes of auditwheel repair, and point to the libmpi
# in mpi4py_mpich package. See patch_libs_for_pip.py and
# https://bodo.atlassian.net/wiki/spaces/DD/pages/946929672/Bodo+Linux+pip+package
# for more information
export PYBIN=/opt/python/cp310-cp310/bin  # patch_libs_for_pip.py looks for $PYBIN
$PYBIN/python buildscripts/bodo/pip/centos7/patch_libs_for_pip.py

# upload with twine to PyPI
cp .pypirc ~/.pypirc
$PYBIN/python -m twine upload -r pypi /bodo/wheelhouse/*.whl
