#! /bin/bash -i
# Note that this script should be run interactively with `bash -i`

set -e pipefail

# Check if GITHUB_TOKEN is in the environment
if [ -z "$GITHUB_TOKEN" ]; then
 echo -n "Enter your GitHub token: "
  read GITHUB_TOKEN
fi

# Check if BRANCH_NAME is in the environment
if [ -z "$BRANCH_NAME" ]; then
  echo -n "Enter a branch name: "
  read BRANCH_NAME
fi

# Clone the repo and checkout the desired branch
psh git clone -b $BRANCH_NAME https://$GITHUB_TOKEN@github.com/Bodo-inc/Bodo.git

# Update conda and install conda-lock
psh sudo /opt/conda/bin/conda update conda --force --yes
psh sudo /opt/conda/bin/mamba install conda-lock -c conda-forge -n base --yes

# Install conda deps
psh conda-lock install --dev --mamba -n DEV ~/Bodo/buildscripts/envs/conda-lock.yml
# Remove conda install mpi to prefer intel MPI on the platform
psh conda run -n DEV conda remove mpi mpich --force --yes

conda activate DEV

cd ~/Bodo
psh env BODO_SKIP_CPP_TESTS=1 USE_BODO_ARROW_FORK=1 pip install --no-deps --no-build-isolation -ve .

cd BodoSQL
psh python -m pip install --no-deps --no-build-isolation -ve .
cd ..

cd iceberg
psh python -m pip install --no-deps --no-build-isolation -ve .
cd ..

cd bodo-platform-image/bodo-platform-utils/
psh pip install -ve .
# Ensure that modify time for bodosql wrapper is the same on all nodes. If it is
# not, then numba may not work as expected.
psh touch -am -t 202401010000 ./bodo_platform_utils/bodosqlwrapper.py
cd ~/Bodo
