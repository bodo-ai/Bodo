#!/usr/bin/env bash
set -euo pipefail

CLUSTER_ID="$(terraform output --json | jq -r '.emr_cluster_id.value')"
REGION="$(terraform output --json | jq -r '.emr_cluster_region.value')"

echo "Fetching EMR steps..."
STEP_IDS=$(aws emr list-steps \
  --cluster-id "$CLUSTER_ID" \
  --region "$REGION" \
  --query 'reverse(Steps)[].Id' \
  --output text)

FAILED=0

for STEP_ID in $STEP_IDS; do
  echo "Waiting for step $STEP_ID..."
  aws emr wait step-complete \
    --cluster-id "$CLUSTER_ID" \
    --step-id "$STEP_ID" \
    --region "$REGION"

  STATUS=$(aws emr describe-step \
    --cluster-id "$CLUSTER_ID" \
    --step-id "$STEP_ID" \
    --region "$REGION" \
    --query 'Step.Status.State' \
    --output text)

  # Sleep to avoid rate limit
  sleep 3

  NAME=$(aws emr describe-step \
    --cluster-id "$CLUSTER_ID" \
    --step-id "$STEP_ID" \
    --region "$REGION" \
    --query 'Step.Name' \
    --output text)

  echo "Step '$NAME' finished with status: $STATUS"

  if [[ "$STATUS" != "COMPLETED" ]]; then
    FAILED=1
  fi

done

if [[ "$FAILED" -eq 1 ]]; then
  echo "One or more steps failed."
  exit 1
else
  echo "All steps completed successfully."
fi
