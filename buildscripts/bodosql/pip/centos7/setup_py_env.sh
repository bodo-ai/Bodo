#!/bin/bash
set -exo pipefail

# We don't support Python 3.6, 3.7 (because of Bodo or a dependency of Bodo)
rm -rf /opt/python/cp36-cp36m
rm -rf /opt/python/cp37-cp37m
rm -rf /opt/python/cp311-cp311

# Install Python packages required to build Bodo pip package. Install for all Python
# versions that we support
yum update -y
# rpm -qa | grep java-1.8.0 | xargs yum -y remove
yum install -y java-11-openjdk-devel
yum install -y maven
alternatives --set java java-11-openjdk.x86_64
alternatives --set javac java-11-openjdk.x86_64
for PYBIN in /opt/python/cp*/bin; do
    # For Python 3.10, the earliest numpy binaries available are 1.21, and source
    # packages prior to 1.21 do not build.
    "${PYBIN}/pip" install bodo
done
