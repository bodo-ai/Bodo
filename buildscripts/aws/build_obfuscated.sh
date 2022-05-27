#!/bin/bash

set -exo pipefail

export PATH=$HOME/miniconda3/bin:$PATH

echo "********** Creating Conda Env **********"
conda create -y -n bodo_build conda-build anaconda-client conda-verify conda=4.12

echo "********** Flake8-ing **********"
conda install -y flake8
flake8 bodo

echo "********** Obfuscating **********"
conda install -y astor -c conda-forge
cd $CODEBUILD_SRC_DIR/obfuscation
./do_obfuscation.py


echo "********** Building Bodo **********"
source activate bodo_build
cd $CODEBUILD_SRC_DIR
export BODO_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`
if [[ "$UseNumbaDev" == "true" ]]; then
    cd $CODEBUILD_SRC_DIR/buildscripts/bodo-numba-dev-conda-recipe/
    echo "Using Numba-Dev Conda Recipe"
else
    cd $CODEBUILD_SRC_DIR/buildscripts/bodo-conda-recipe/
fi
export CHECK_LICENSE_EXPIRED="0"
export CHECK_LICENSE_CORE_COUNT="0"
if [[ "$UseNumbaDev" == "true" ]]; then
    conda-build . -c numba/label/dev -c conda-forge --no-test
    echo "Using numba/label/dev channel"
else
    conda-build . -c conda-forge --no-test
fi
