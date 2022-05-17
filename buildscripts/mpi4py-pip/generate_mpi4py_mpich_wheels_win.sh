#!/bin/bash
set -e -u -x

# See https://bodo.atlassian.net/wiki/spaces/DD/pages/972390401/Windows+pip+package
# for more information

# On Azure CI, all of the prerequisites and environment are already set up
# through the VM image and our Azure yml build scripts
# For non-CI:
# - Install Visual Studio Build Tools (I tested with version 2019)
# - Install Anaconda or miniconda (don't add conda to PATH during install)
# - Install Git Bash (don't add anything to PATH during install)
# Add conda shell script to your ~/.bashrc, for example as described here:
# https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473
# Open Git Bash
# $ conda activate
# If Visual Studio tools are not in PATH, run this on the command line:
# "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# Now you can run this script

# XXX set these for your environment:
# configure shell for conda activate
. /path/to/conda/etc/profile.d/conda.sh
export PATH_TO_BODO_SRC=/path/to/bodo_src
cd some_working_directory

# install MS-MPI (note that this happens in base conda environment)
conda install msmpi -c conda-forge -y

git clone --depth 1 --branch 3.1.2 https://github.com/mpi4py/mpi4py.git
cd mpi4py
cp -r $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/mpiexec .
mkdir mpiexec/.libs
# copy MS-MPI mpiexec binaries to mpi4py
cp $CONDA_PREFIX/Library/bin/mpiexec.exe mpiexec
cp $CONDA_PREFIX/Library/bin/smpd.exe mpiexec
cp $CONDA_PREFIX/Library/bin/msmpi.dll mpiexec/.libs
# patch mpi4py
git apply $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/patch-3.1.2.diff
# append MS-MPI license to mpi4py license
cat $PATH_TO_BODO_SRC/buildscripts/mpi4py-pip/MS-MPI_license.txt >> LICENSE.rst

for PYTHON_VER in "3.8" "3.9" "3.10"
do
    conda create -n BUILDPIP python=$PYTHON_VER msmpi -c conda-forge -y
    conda activate BUILDPIP
    export MSMPI_BIN=$CONDA_PREFIX/Library/bin
    export MSMPI_INC=$CONDA_PREFIX/Library/include
    export MSMPI_LIB64=$CONDA_PREFIX/Library/lib
    export MSMPI_LIB32=""
    python setup.py bdist_wheel
    conda deactivate
    conda env remove -n BUILDPIP
done

# upload with twine to PyPI
conda install twine -y
python -m twine upload -r pypi dist/*.whl
