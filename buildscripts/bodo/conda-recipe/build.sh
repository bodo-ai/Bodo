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

export SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION"
export CMAKE_GENERATOR='Ninja'

# Build the wheel. We can use this for only-Pip wheel builds
$PYTHON -m pip install \
    --no-deps --no-build-isolation -vv \
    --config-settings=build.verbose=true \
    --config-settings=logging.level="DEBUG" \
    --config-settings=cmake.args="-DCMAKE_INSTALL_PREFIX=$PREFIX;-DCMAKE_INSTALL_LIBDIR=lib;-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY;-DCMAKE_OSX_SYSROOT=/opt/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk" \
    .

sccache --show-stats
