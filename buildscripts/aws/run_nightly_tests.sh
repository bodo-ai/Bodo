#!/bin/bash
set -exo pipefail

USERNAME=`credstash -r us-east-2 get artifactory.ci.username`
TOKEN=`credstash -r us-east-2 get artifactory.ci.token`

export PATH=$HOME/miniconda3/bin:$PATH
conda create -y -n bodo_dev
source activate bodo_dev

conda install -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/bodo-binary bodo-2020.09-py38hc547734_19.tar.bz2

python testitout.py
# pip install pytest
