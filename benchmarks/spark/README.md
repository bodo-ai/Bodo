# Spark NYC Taxi Precipitation Benchmark on EMR

---

## Prerequisites

1. **AWS CLI**: Installed and configured with access keys.
2. **Terraform**: Installed on your local machine.
3. **jq**: Installed on your local machine.
4. **gzip**: Installed on your local machine.

---

## Steps to Use the Terraform Script

### 1. Enter the bechmarks/spark Directory

### 2. Initialize Terraform

Run the following command to initialize the Terraform project:

```bash
terraform init
```

### 3. Deploy the Infrastructure

Apply the Terraform script to deploy the resources:

```bash
terraform apply
```

You will be prompted to confirm the deployment. Type `yes` to proceed.

### 4. Verify Deployment

After the deployment completes, the EMR cluster and associated resources will be provisioned. The Python script will be uploaded to a created S3 bucket.

---

## Running the Python Script

The script will automatically run on the EMR cluster as a step during the deployment.

---

## Wait for the Script to Complete

Run the following command to wait for the script to complete:

```bash
./wait_for_steps.sh
```

This will take a few minutes.

---

## Retrieving Logs

### 2. Download Logs Using AWS CLI

Use the AWS CLI to download the logs:

```bash
aws s3 cp s3://"$(terraform output --json | jq -r '.s3_bucket_id.value')"/logs/"$(terraform output --json | jq -r '.emr_cluster_id.value')" ./emr-logs --recursive --region "$(terraform output --json | jq -r '.emr_cluster_region.value')"
```

### 3. Explore the Logs

Logs are structured into the following directories:

- `steps/`: Logs for each step.
- `node/`: Logs for individual cluster nodes.
- `applications/`: Logs for applications like Spark.

Example command to view step logs with execution time result:

```bash
gzip -d ./emr-logs/steps/*/*
cat ./emr-logs/steps/*/stdout
```

---

## Cleaning Up

To destroy the deployed resources and avoid incurring additional costs, run:

```bash
terraform destroy
```

You will be prompted to confirm. Type `yes` to proceed.
