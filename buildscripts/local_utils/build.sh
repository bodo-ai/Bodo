#!/bin/bash

#Builds all of the Bodo and Bodo's sub-components
#TODO add options to avoid building iceberg/BodoSQL if not needed
pip install --no-deps --no-build-isolation -ve .; cd iceberg; pip install -v --no-deps --no-build-isolation .; cd ../BodoSQL; pip install -v --no-deps --no-build-isolation .; cd ..;
