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

# configure shell for conda activate
. `cygpath -u $CONDA`/etc/profile.d/conda.sh
conda activate

# obfuscate
cd obfuscation
pip install astor
python do_obfuscation.py

# rename to pyx
cd ..
python rename_to_pyx.py

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

# build Bodo wheels
export BUILD_PIP=1
for PYTHON_VER in "3.8" "3.9"
do
    conda create -n BUILDPIP python=$PYTHON_VER msmpi boost-cpp -c conda-forge -y
    conda activate BUILDPIP
    python -m pip install Cython numpy==1.18.* wheel pyarrow==5.0.0 mpi4py_mpich>3.0.3
    # copy SSL DLLs to bodo source directory to bundle them in package
    cp `cygpath -u $CONDA_PREFIX`/Library/bin/libssl-*-x64.dll bodo/libs
    cp `cygpath -u $CONDA_PREFIX`/Library/bin/libcrypto-*-x64.dll bodo/libs
    python setup.py bdist_wheel
    conda deactivate
    conda env remove -n BUILDPIP
done

# upload with twine to PyPI
cp .pypirc ~/.pypirc
conda install twine -y
python -m twine upload -r pypi dist/*.whl
