#!/bin/bash

set -e  # exit immediately on error

# === CONFIG ===
SPARK_VERSION="3.5.6"
ICEBERG_VERSION="1.4.2"
SPARK_DIR="$HOME/spark-${SPARK_VERSION}"
SPARK_TGZ="spark-${SPARK_VERSION}-bin-hadoop3.tgz"
SPARK_URL="https://downloads.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_TGZ}"

ICEBERG_JAR_NAME="iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar"
ICEBERG_JAR_URL="https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/${ICEBERG_JAR_NAME}"
ICEBERG_JAR_DIR="$SPARK_DIR/jars"

# ==============

echo "Removing any old Spark install at $SPARK_DIR..."
rm -rf "$SPARK_DIR"

echo "Downloading Spark ${SPARK_VERSION}..."
curl -fLO "$SPARK_URL"

echo "Extracting Spark..."
tar -xzf "$SPARK_TGZ"
mv "spark-${SPARK_VERSION}-bin-hadoop3" "$SPARK_DIR"
rm "$SPARK_TGZ"

echo "Spark installed to $SPARK_DIR"

echo "Creating JAR directory at $ICEBERG_JAR_DIR"
mkdir -p "$ICEBERG_JAR_DIR"

echo "Downloading Iceberg runtime JAR..."
curl -fLo "$ICEBERG_JAR_DIR/$ICEBERG_JAR_NAME" "$ICEBERG_JAR_URL"

echo "Iceberg JAR downloaded to $ICEBERG_JAR_DIR/$ICEBERG_JAR_NAME"

echo "Verifying spark-submit..."
"$SPARK_DIR/bin/spark-submit" --version

echo ""
echo "Done!"
echo "To run Spark with Iceberg:"
echo ""
echo "  $SPARK_DIR/bin/spark-submit \\"
echo "    --jars $ICEBERG_JAR_DIR/$ICEBERG_JAR_NAME \\"
echo "    your_script.py"
