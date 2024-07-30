
#! /bin/bash
set -e pipefail
echo -n "Enter your GitHub token: "
read GITHUB_TOKEN 
echo -n "Enter a branch name: "
read BRANCH_NAME
git clone https://$GITHUB_TOKEN@github.com/Bodo-inc/Bodo.git
cd ~/Bodo
git checkout $BRANCH_NAME
sudo /opt/conda/bin/conda update conda --force --yes
sudo /opt/conda/bin/mamba install conda-lock -c conda-forge -n base --yes
cd buildscripts/envs
conda-lock install --dev --mamba -n DEV conda-lock.yml
conda run -n DEV conda remove mpi mpich --force --yes
cd ~/Bodo
conda run -n DEV pip install --no-deps --no-build-isolation -ve .
cd BodoSQL && conda run -n DEV python setup.py develop && cd ..
cd iceberg && conda run -n DEV python setup.py develop && cd ..
cd bodo-platform-image/bodo-platform-utils/ && conda run -n DEV pip install -ve . && cd ../..
