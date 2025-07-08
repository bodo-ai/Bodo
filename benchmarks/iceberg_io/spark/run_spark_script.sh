#!/bin/bash

# === 2. Load Spark config ===
source ./config_spark_env.sh

# === 3. Run Spark job ===
echo "Running Spark job..."

"$SPARK_DIR/bin/spark-submit" \
  --jars "$(echo $JAR_DIR/*.jar | tr ' ' ',')" \
  --conf "spark.sql.catalog.$CATALOG_NAME=org.apache.iceberg.spark.SparkCatalog" \
  --conf "spark.sql.catalog.$CATALOG_NAME.catalog-impl=software.amazon.s3tables.iceberg.S3TablesCatalog" \
  --conf "spark.sql.catalog.$CATALOG_NAME.warehouse=$WAREHOUSE_ARN" \
  "$PY_SCRIPT"