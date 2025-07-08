#!/bin/bash

# === 1. Define Spark Catalog Config ===

# You can source this from a .env file or parameterize later
export SPARK_DIR="$HOME/spark-3.5.6"
export JAR_DIR="./jars"
export PY_SCRIPT="spark_iceberg_benchmark.py"
export CATALOG_NAME="s3tbl"
export WAREHOUSE_ARN="arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
