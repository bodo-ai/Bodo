#!/bin/sh
set -euo pipefail

# URLs
MPICH_TARBALL="mpich-${MPICH_VERSION}.tar.gz"
MPICH_URL="https://www.mpich.org/static/downloads/${MPICH_VERSION}/${MPICH_TARBALL}"

# Directories
BUILD_DIR="${PWD}/mpich-${MPICH_VERSION}"
curl -LO "$MPICH_URL"
tar -xzf "$MPICH_TARBALL"

echo "BUILD_DIR: $BUILD_DIR"
echo "MPICH_INSTALL_DIR: $MPICH_INSTALL_DIR"

cd "$BUILD_DIR"
./configure \
    --prefix="${MPICH_INSTALL_DIR}" \
    --disable-fortran \
    --disable-cxx \
    --disable-doc \
    --disable-dependency-tracking \
    --disable-static

make

make install
