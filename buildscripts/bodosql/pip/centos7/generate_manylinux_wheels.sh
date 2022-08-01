#! /bin/bash
set -e -u -x

cd /BodoSQL/bodosql

export PYBIN=/opt/python/cp39-cp39/bin

# build Bodo wheels
${PYBIN}/python setup.py bdist_wheel

# Install twine.
# TODO: Move to the image.
${PYBIN}/python -m pip install twine

# upload with twine to PyPI
cp .pypirc ~/.pypirc
$PYBIN/python -m twine upload -r pypi /bodosql/dist/*.whl
