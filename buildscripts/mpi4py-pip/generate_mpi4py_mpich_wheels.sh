#!/bin/bash
set -exo pipefail

# See https://bodo.atlassian.net/wiki/spaces/DD/pages/946929672/Bodo+Linux+pip+package
# for more information

cd /
git clone --depth 1 --branch 3.1.2 https://github.com/mpi4py/mpi4py.git
cd mpi4py

cp -r /bodo/buildscripts/mpi4py-pip/mpiexec .
# copy MPICH mpiexec binaries to mpi4py
cp /mpich/bin/mpiexec mpiexec
cp /mpich/bin/hydra_pmi_proxy mpiexec
# patch mpi4py (mainly for mpiexec script)
git apply /bodo/buildscripts/mpi4py-pip/patch-3.1.2.diff
# append MPICH license to mpi4py license
cat /bodo/buildscripts/mpi4py-pip/mpich_COPYRIGHT.txt >> LICENSE.rst

# build mpi4py_mpich wheels
for PYBIN in /opt/python/cp*/bin; do
    MPICC=/mpich/bin/mpicc "${PYBIN}/python" setup.py bdist_wheel
done

# run auditwheel repair to include libmpi in package and make the mpi4py
# binaries point to it
for WHL in dist/mpi4py_mpich*whl; do
    auditwheel repair --plat manylinux2014_x86_64 $WHL
done

# upload with twine to PyPI
/opt/python/cp39-cp39/bin/python -m twine upload -r pypi wheelhouse/*.whl
