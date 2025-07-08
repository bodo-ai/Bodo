#!/bin/bash

# Required JARs
required_jars=(
  "aws-core-2.29.26.jar"
  "utils-2.29.26.jar"
  "regions-2.29.26.jar"
  "auth-2.29.26.jar"
  "s3-2.29.26.jar"
  "arns-2.29.26.jar"
  "iceberg-spark-runtime-3.5_2.12-1.6.1.jar"
  "s3-tables-catalog-for-iceberg-runtime-0.1.7.jar"
)

missing=0

echo "Checking required JARs in ./jars/..."
for jar in "${required_jars[@]}"; do
  if [[ ! -f "jars/$jar" ]]; then
    echo "MISSING: jars/$jar"
    missing=1
  else
    echo "FOUND:   jars/$jar"
  fi
done

if [[ $missing -eq 0 ]]; then
  echo -e "\nAll required JARs are present."
else
  echo -e "\nSome required JARs are missing. Re-run your curl download script or manually fetch them."
fi
