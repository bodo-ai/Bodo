#!/bin/bash
set -e  # Exit immediately on any error

echo "Step 1: Configuring Spark environment..."
source ./config_spark_env.sh

echo "Step 2: Installing Iceberg + AWS JARs..."
bash ./install_spark_iceberg.sh

echo "Step 3: Setting up Amazon S3Tables Catalog..."
bash ./setup_s3tables.sh

echo "Step 4: Checking that required JARs are present..."
bash ./check_jars.sh
