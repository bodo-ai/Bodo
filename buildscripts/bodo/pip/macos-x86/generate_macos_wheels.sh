#!/bin/bash
set -exo pipefail

# obfuscate
cd obfuscation
pip3 install astor
python3 do_obfuscation.py
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

# Download MPICH for linking
MY_DIR=`pwd`
curl https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz --output mpich-3.3.2.tar.gz
tar xf mpich-3.3.2.tar.gz
mkdir mpich ; cd mpich-3.3.2 ; ./configure --prefix=$MY_DIR/mpich --enable-fast --enable-shared --disable-fortran --disable-opencl --disable-cxx --with-device=ch3
make ; make install
cd ..

# Create the wheelhouse directory for each wheel.
mkdir -p wheelhouse

# build Bodo wheels
for PYTHON_VER in "3.10" "3.11" "3.12"
do
    conda create -n BUILDPIP python=$PYTHON_VER boost-cpp=1.85 -c conda-forge -y
    source activate BUILDPIP
    # For Python 3.10, the earliest numpy binaries available are 1.21, and source
    # packages prior to 1.21 do not build.
    if python -c 'import sys; sys.exit(sys.version_info[:2] >= (3, 10))'; then
        # exit code 0 when version < 3.10
        python -m pip install Cython "numpy==1.18.*" wheel pyarrow==16.1.0 mpi4py_mpich==3.1.2
    else
        # exit code 1 when version >= 3.10
        python -m pip install Cython "numpy==1.21.*" wheel pyarrow==16.1.0 mpi4py_mpich==3.1.2
    fi
    PYARROW_PATH=`python -c "import pyarrow; print('/'.join(pyarrow.__file__.split('/')[:-1]))"`
    ln -s $PYARROW_PATH/libarrow.1500.dylib        $PYARROW_PATH/libarrow.so
    ln -s $PYARROW_PATH/libparquet.1500.dylib      $PYARROW_PATH/libparquet.so
    # Bundle libssl and libcrypto to package.
    cp $CONDA_PREFIX/lib/libssl*.dylib bodo/libs/.
    cp $CONDA_PREFIX/lib/libcrypto*.dylib bodo/libs/.
    CFLAGS='-I'$MY_DIR'/mpich/include -I'$PYARROW_PATH'/include -L'$MY_DIR'/mpich/lib -L'$PYARROW_PATH' -Wl,-rpath,'$MY_DIR'/mpich/lib -Wl,-rpath,'$PYARROW_PATH'' python setup.py bdist_wheel
    mv dist/*.whl wheelhouse/.
    conda deactivate
    conda env remove -n BUILDPIP
done


# Need to update the library paths. See patch_libs_for_pip.py and
# https://bodo.atlassian.net/wiki/spaces/DD/pages/951058433/macOS+pip+package
# for more information
python3 buildscripts/bodo/pip/macos-x86/patch_libs_for_pip.py


# upload with twine to PyPI
cp .pypirc ~/.pypirc
python3 -m pip install twine
python3 -m twine upload -r pypi wheelhouse/*.whl
