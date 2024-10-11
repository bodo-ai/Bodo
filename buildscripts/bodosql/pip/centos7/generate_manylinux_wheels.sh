#! /bin/bash
set -e -u -x

cd bodosql/BodoSQL

export PYBIN=/opt/python/cp39-cp39/bin

# build Bodo wheels
${PYBIN}/python -m pip wheel \
    --wheel-dir=/tmp/wheelhouse \
    --no-deps --no-build-isolation -vv .

# Install twine.
# TODO: Move to the image.
${PYBIN}/python -m pip install twine

# upload with twine to PyPI
cp .pypirc ~/.pypirc
$PYBIN/python -m twine upload -r pypi /tmp/wheelhouse/*.whl
