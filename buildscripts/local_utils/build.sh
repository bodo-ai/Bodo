#!/bin/bash

#Builds all of the Bodo and Bodo's sub-components
#TODO add options to avoid building iceberg/BodoSQL if not needed
pip install --no-deps --no-build-isolation -ve .; cd iceberg; python setup.py develop; cd ../BodoSQL; python setup.py develop; cd ..;
