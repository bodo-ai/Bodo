#!/bin/bash

source /home/bodo/.bashrc

# Start uvicorn
exec /opt/conda/bin/uvicorn main:app --host 0.0.0.0 --port 8888
