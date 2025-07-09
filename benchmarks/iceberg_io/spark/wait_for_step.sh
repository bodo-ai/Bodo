# Extract cluster and step IDs from Terraform JSON output
CLUSTER_ID="$(terraform output --json | jq -r '.emr_cluster_id.value')"
EMR_CLUSTER_REGION="$(terraform output --json | jq -r '.emr_cluster_region.value')"
STEP_ID0="$(aws emr list-steps --cluster-id "$CLUSTER_ID" --query 'Steps[0].Id' --output text --region "$EMR_CLUSTER_REGION")"
# TODO: Reenable when doing full run.
# STEP_ID1="$(aws emr list-steps --cluster-id "$CLUSTER_ID" --query 'Steps[1].Id' --output text --region "$EMR_CLUSTER_REGION")"
# STEP_ID2="$(aws emr list-steps --cluster-id "$CLUSTER_ID" --query 'Steps[2].Id' --output text --region "$EMR_CLUSTER_REGION")"

# Wait for the last EMR step to complete
aws emr wait step-complete --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID0" --region "$EMR_CLUSTER_REGION"
# aws emr wait step-complete --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID1" --region "$EMR_CLUSTER_REGION"
# aws emr wait step-complete --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID2" --region "$EMR_CLUSTER_REGION"

# Check the step status
STEP_STATUS=$(aws emr describe-step --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID0" --query 'Step.Status.State' --output text --region "$EMR_CLUSTER_REGION")

if [ "$STEP_STATUS" == "COMPLETED" ]; then
    echo "Step completed successfully."
else
    echo "Step did not complete successfully. Status: $STEP_STATUS"
fi

