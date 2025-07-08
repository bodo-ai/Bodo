#!/bin/bash

# Use Spark 3.5.6 only for this project
SPARK_HOME="$HOME/spark-3.5.6"
ICEBERG_JAR="./jars/iceberg-spark-runtime-3.5_2.12-1.4.2.jar"

"$SPARK_HOME/bin/spark-submit" \
  --jars "$ICEBERG_JAR" \
  "$@"