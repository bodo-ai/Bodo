set -exo pipefail

# Remove unused files to make the docker images smaller
export PATH=$HOME/miniconda3/bin:$PATH
conda clean -a -y
