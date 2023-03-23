#!/bin/bash
set -eo pipefail
set +x

# Used to run the maven unit tests inside AWS codebuild

# Deactivate env in case this was called by another file that
# activated the env
source deactivate || true
export PATH=$HOME/mambaforge/bin:$PATH
source activate $CONDA_ENV

wget https://download.oracle.com/otn_software/linux/instantclient/215000/instantclient-basic-linux.x64-21.5.0.0.0dbru.zip
unzip instantclient-basic-linux.x64-21.5.0.0.0dbru.zip -d /usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib/instantclient_21_5:$LD_LIBRARY_PATH


# run the maven tests
cd BodoSQL/calcite_sql; mvn '-Dtest=com.bodosql.calcite.application.*Test' test; cd ../..
