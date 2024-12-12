# Extract cluster and step IDs from Terraform JSON output
CLUSTER_ID="$(terraform output --json | jq -r '.emr_cluster_id.value')"
EMR_CLUSTER_REGION="$(terraform output --json | jq -r '.emr_cluster_region.value')"
STEP_ID="$(aws emr list-steps --cluster-id "$CLUSTER_ID" --query 'Steps[0].Id' --output text --region "$EMR_CLUSTER_REGION")"

# Wait for the EMR step to complete
aws emr wait step-complete --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID" --region "$EMR_CLUSTER_REGION"

# Check the step status
STEP_STATUS=$(aws emr describe-step --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID" --query 'Step.Status.State' --output text --region "$EMR_CLUSTER_REGION")

if [ "$STEP_STATUS" == "COMPLETED" ]; then
    echo "Step completed successfully."
else
    echo "Step did not complete successfully. Status: $STEP_STATUS"
fi

