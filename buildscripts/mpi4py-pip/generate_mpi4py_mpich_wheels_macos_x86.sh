#!/bin/bash
set -e -u -x

# This script assumes you have already built mpich locally.
# See https://bodo.atlassian.net/wiki/spaces/DD/pages/951058433/macOS+pip+package
# for more information

git clone --depth 1 --branch 3.1.2 https://github.com/mpi4py/mpi4py.git
cd mpi4py

cp -r $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/mpiexec .
# copy MPICH mpiexec binaries to mpi4py
cp $PATH_TO_MPICH/mpich/bin/mpiexec mpiexec
cp $PATH_TO_MPICH/mpich/bin/hydra_pmi_proxy mpiexec
# patch mpi4py (mainly for mpiexec script)
git apply $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/patch-3.1.2.diff
# append MPICH license to mpi4py license
cat $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/mpich_COPYRIGHT.txt >> LICENSE.rst

for PYTHON_VER in "3.8" "3.9"
do
    conda create -n BUILDPIP python=$PYTHON_VER -c conda-forge -y
    conda activate BUILDPIP
    MPICC=$PATH_TO_MPICH/mpich/bin/mpicc
    python setup.py bdist_wheel
    conda deactivate
    conda env remove -n BUILDPIP
done

# run delocate to include libmpi in package and make the mpi4py
# binaries point to it
# https://pypi.org/project/delocate/
for WHL in dist/mpi4py_mpich*whl; do
    delocate-wheel $WHL
done

# upload with twine to PyPI
python3 -m pip install twine
python3 -m twine upload -r pypi dist/*.whl
