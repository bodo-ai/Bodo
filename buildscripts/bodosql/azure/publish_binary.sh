#!/bin/bash
set -exo pipefail

export PATH=$HOME/miniconda3/bin:$PATH
source activate $CONDA_ENV

# Upload to artifactory
conda install -y conda-build anaconda-client conda-verify curl -c conda-forge

BODOSQL_CHANNEL_NAME=${1:-bodo-binary}

# Note: It is difficult to enforce the exact version of bodo to use for the build, outside of
# modifying the build recipe. Therefore, if we update BodoSQL's build process to require a specific version
# of Bodo, I'm requiring that we update the recipe in buildscripts/bodosql/conda-recipe/meta.yaml,
# and do a mini-release.
BODO_CHANNEL_NAME="bodo.ai"

echo "********** Publishing to Artifactory **********"
USERNAME=`cat $HOME/secret_file | grep artifactory.ci.username | cut -f 2 -d' '`
TOKEN=`cat $HOME/secret_file | grep artifactory.ci.token | cut -f 2 -d' '`
ANACONDA_TOKEN=`cat $HOME/secret_file | grep anaconda.org.token | cut -f 2 -d' '`

# Get the BodoSQL version
export BODOSQL_VERSION=`python -c "import versioneer; print(versioneer.get_version())"`
export IS_RELEASE=`git tag --points-at HEAD`

# We follow the following convention for release:
# Major Release: Original month release, i.e. 2021.8b1 or 2021.11
# There should always be 1 dot, even in beta.
# Minor Release: Repeat release in a month, there should be a second dot
# i.e. 2021.8.1beta or 2021.11.2
# Release Candidate: If we need to test a release before the major release
# we will create a minor release ending in .0 and append rc#
# i.e. 2021.9.0betarc1 or 2021.11.0rc2
# For more information, please see our confluence doc: https://bodo.atlassian.net/wiki/spaces/B/pages/1020592198/Release+Checklist
IS_MAJOR_RELEASE=0
if [[ -n "$IS_RELEASE" ]]; then
    IS_MAJOR_RELEASE=`python -c "import os; print(int(len(os.environ[\"BODOSQL_VERSION\"].split(\".\")) < 3))"`
fi
label=""
if [[ "$IS_MAJOR_RELEASE" == 1 ]] && [[ "$BODOSQL_CHANNEL_NAME" == "bodo.ai" ]]; then
    # If we have a major release upload with our main anaconda label
    label="main"
elif [[ "$BODOSQL_CHANNEL_NAME" == "bodo.ai" ]] && [[ -n "$IS_RELEASE" ]]; then
    # If we have a minor release upload with our dev anaconda label
    label="dev"
fi

conda-build buildscripts/bodosql/conda-recipe -c https://${USERNAME}:${TOKEN}@bodo.jfrog.io/artifactory/api/conda/$BODO_CHANNEL_NAME -c conda-forge --no-test

for package in `ls $CONDA_PREFIX/conda-bld/noarch/bodosql*.tar.bz2`; do
    package_name=`basename $package`
    curl -u${USERNAME}:${TOKEN} -T $package "https://bodo.jfrog.io/artifactory/${BODOSQL_CHANNEL_NAME}/noarch/$package_name"
    if [[ ! -z "$label" ]]; then
        # `--skip-existing` skips the upload in case the package already exists.
        # Since the pipeline runs every night, we don't want to replace
        # the package on anaconda and accidentally break something.
        anaconda -t $ANACONDA_TOKEN upload -u bodo.ai -c bodo.ai $package --label $label --skip-existing
    fi
done

# reindex conda
curl -X POST https://$USERNAME:$TOKEN@bodo.jfrog.io/artifactory/api/conda/$BODOSQL_CHANNEL_NAME/reindex
