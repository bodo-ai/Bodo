#!/bin/bash

set -e

# === Configuration ===
SPARK_VERSION="3.5.6"
SPARK_DIR="$HOME/spark-$SPARK_VERSION"
SPARK_TGZ="spark-$SPARK_VERSION-bin-hadoop3.tgz"
SPARK_URL="https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/$SPARK_TGZ"

ICEBERG_VERSION="1.6.1"
AWS_SDK_VERSION="2.29.26"
S3TABLES_VERSION="0.1.7"

WAREHOUSE_ARN="arn:aws:s3tables:us-east-2:427443013497:bucket/tpch"
CATALOG_NAME="s3tbl"

PY_SCRIPT="spark_iceberg_benchmark.py"
JAR_DIR="jars"

# === 1. Download Spark ===
if [ ! -d "$SPARK_DIR" ]; then
  echo "Downloading Spark $SPARK_VERSION..."
  curl -LO "$SPARK_URL"
  tar -xzf "$SPARK_TGZ"
  mv spark-$SPARK_VERSION-bin-hadoop3 "$SPARK_DIR"
  rm "$SPARK_TGZ"
fi

# === 2. Download Required JARs ===
mkdir -p "$JAR_DIR"
cd "$JAR_DIR"

echo "Downloading AWS SDK dependencies..."
for jar in aws-core utils regions auth s3 arns; do
  curl -s -LO "https://repo1.maven.org/maven2/software/amazon/awssdk/$jar/$AWS_SDK_VERSION/$jar-$AWS_SDK_VERSION.jar" \
    || echo "Failed to get $jar"
done

echo "Downloading Iceberg Spark runtime..."
curl -s -LO "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/$ICEBERG_VERSION/iceberg-spark-runtime-3.5_2.12-$ICEBERG_VERSION.jar"

echo "Downloading Amazon S3Tables Catalog runtime..."
curl -s -LO "https://repo1.maven.org/maven2/software/amazon/s3tables/s3-tables-catalog-for-iceberg-runtime/$S3TABLES_VERSION/s3-tables-catalog-for-iceberg-runtime-$S3TABLES_VERSION.jar"

cd ..

