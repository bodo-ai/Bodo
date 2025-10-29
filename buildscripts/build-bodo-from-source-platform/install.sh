#! /bin/bash -i
# Note that this script should be run interactively with `bash -i`

set -e pipefail

# Check if GITHUB_TOKEN is in the environment
if [ -z "$GITHUB_TOKEN" ]; then
 echo -n "Enter your GitHub token: "
  read GITHUB_TOKEN
fi

# Clone the repo and checkout the desired branch
psh git clone -b main https://$GITHUB_TOKEN@github.com/bodo-ai/Bodo.git

# Install Pixi
psh bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
source ~/.bashrc

# Install development deps
cd ~/Bodo
psh pixi install -e platform-dev
# Remove conda install mpi to prefer intel MPI on the platform
psh env BODO_SKIP_CPP_TESTS=1 pixi run build -e platform-dev

pixi shell -e platform-dev
cd bodo-platform-image/bodo-platform-utils/
psh pip install -ve .
# Ensure that modify time for bodosql wrapper is the same on all nodes. If it is
# not, then numba may not work as expected.
psh touch -am -t 202401010000 ./bodo_platform_utils/bodosqlwrapper.py
cd ~/Bodo
