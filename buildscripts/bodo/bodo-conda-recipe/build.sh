# LDSHARED="mpicxx -cxx=$GXX -shared" LD="mpicxx -cxx=$GXX" \
# CC="mpicxx -cxx=$GXX -std=c++11" GXX="mpicxx -cxx=$GXX -std=c++11" \
# OPENCV_DIR="${PREFIX}" DAALROOT="${PREFIX}"

if  [[ -z ${DISABLE_SCCACHE_IN_BUILD+x} ]]; then
    # Enable SCCache
    export SCCACHE_BUCKET=engine-codebuild-cache
    export SCCACHE_REGION=us-east-2
    export SCCACHE_S3_USE_SSL=true
    export SCCACHE_S3_SERVER_SIDE_ENCRYPTION=true
fi


# Conda-Build deletes .git, so we need to determine the version beforehand
# and then pass it to setuptools_scm in the build script (setup.py).
# https://conda-forge.org/docs/maintainer/knowledge_base.html#using-setuptools-scm
export SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION"

export CONDA_PREFIX_OLD=$CONDA_PREFIX
export CONDA_PREFIX=$PREFIX
export CMAKE_GENERATOR='Ninja'


# Build the wheel. We can use this for only-Pip wheel builds
# TODO: Are all of these necessary?
MACOSX_DEPLOYMENT_TARGET=10.15 \
$PYTHON -m pip wheel \
    --wheel-dir=/tmp/wheelhouse \
    --no-deps --no-build-isolation -vv \
    -Ccmake.verbose=true \
    -Clogging.level="DEBUG" \
    -Ccmake.args="-DCMAKE_C_COMPILER=$CC;-DCMAKE_CXX_COMPILER=$CXX;-DCMAKE_INSTALL_PREFIX=$PREFIX;-DCMAKE_INSTALL_LIBDIR=lib;-DCMAKE_FIND_ROOT_PATH='$PREFIX;$CONDA_PREFIX_OLD/x86_64-conda-linux-gnu/sysroot';-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY" \
    .

sccache --show-stats

# Install the wheel
$PYTHON -m pip install --no-index --find-links=/tmp/wheelhouse bodo
