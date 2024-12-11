# Spark NYC Taxi Precipitation Benchmark on EMR

---

## Prerequisites

1. **AWS CLI**: Installed and configured with access keys.
2. **Terraform**: Installed on your local machine.

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

## Retrieving Logs

### 2. Download Logs Using AWS CLI

Use the AWS CLI to download the logs:

```bash
aws s3 cp s3://spark-benchmark-python-script-bucket-<random characters>/logs/ ./emr-logs --recursive
```

The bucket name can be found in the logs of `terraform apply`.

### 3. Explore the Logs

Logs are structured into the following directories:

- `steps/`: Logs for each step.
- `node/`: Logs for individual cluster nodes.
- `applications/`: Logs for applications like Spark.

Example command to view step logs:

---

## Cleaning Up

To destroy the deployed resources and avoid incurring additional costs, run:

```bash
terraform destroy
```

You will be prompted to confirm. Type `yes` to proceed.

---
