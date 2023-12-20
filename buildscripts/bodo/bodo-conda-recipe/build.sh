# LDSHARED="mpicxx -cxx=$GXX -shared" LD="mpicxx -cxx=$GXX" \
# CC="mpicxx -cxx=$GXX -std=c++11" GXX="mpicxx -cxx=$GXX -std=c++11" \
# OPENCV_DIR="${PREFIX}" DAALROOT="${PREFIX}"

# Conda-Build deletes .git, so we need to determine the version beforehand
# and then pass it to setuptools_scm in the build script (setup.py).
# https://conda-forge.org/docs/maintainer/knowledge_base.html#using-setuptools-scm
export SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION"

MACOSX_DEPLOYMENT_TARGET=10.15 \
$PYTHON setup.py build install --single-version-externally-managed --record=record.txt
