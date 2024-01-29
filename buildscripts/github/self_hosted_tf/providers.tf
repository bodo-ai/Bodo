provider "aws" {
  region = local.aws_region

  default_tags {
    tags = {
      Example = local.prefix
    }
  }
}

# Configure an S3 Bucket for Terraform State
# including Versions, Encryption, Access, and Locking
# Save State to S3 for Remote Development
terraform {
  backend "s3" {
    # Doesn't Support Local Variables
    # Must be same as local.state_bucket
    bucket = "self-hosted-github-actions-tf"
    key    = "terraform.tfstate"
    # Must be same as local.aws_region
    region = "us-east-2"
    # Must be same as local.state_db
    dynamodb_table = "self-hosted-github-actions-tf-locks"
    encrypt        = true
  }
}
