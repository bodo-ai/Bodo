provider "aws" {
  region = "us-east-2"
}

variable "data_folder" {
  description = "The S3 folder containing the TPCH parquet data."
  type        = string
  default     = "s3://bodo-example-data/tpch/SF1/"
}

variable "scale_factor" {
  description = "The scale factor of the TPCH dataset."
  type        = number
  default     = 1
}

variable "queries" {
  description = "List of TPCH queries to run."
  type        = list(number)
  default     = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
}

locals {
  tpch_bucket = regex("^s3://([^/]+)/?", var.data_folder)[0]
}

resource "aws_s3_bucket" "emr_bucket" {
  bucket        = "spark-benchmark-python-script-bucket-${random_id.bucket_id.hex}"
  force_destroy = true
}

resource "random_id" "bucket_id" {
  byte_length = 8
}

resource "aws_s3_object" "python_script" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "scripts/pyspark_queries.py"
  source = "./pyspark_queries.py"
}

resource "aws_s3_object" "bootstrap_script" {
  bucket  = aws_s3_bucket.emr_bucket.id
  key     = "scripts/bootstrap.sh"
  content = <<EOF
#!/bin/bash
sudo pip install -U pandas numpy==1.26.4 pyarrow
EOF
}

resource "aws_emr_cluster" "emr_cluster" {
  name          = "EMR-Cluster"
  release_label = "emr-7.5.0"
  applications  = ["Hadoop", "Spark"]

  ec2_attributes {
    instance_profile                  = aws_iam_instance_profile.emr_profile.arn
    subnet_id                         = aws_subnet.emr_subnet.id
    emr_managed_master_security_group = aws_security_group.allow_access.id
    emr_managed_slave_security_group  = aws_security_group.allow_access.id
  }

  master_instance_group {
    instance_type = "r6i.8xlarge"
  }

  core_instance_group {
    instance_type  = "r6i.8xlarge"
    instance_count = 2

    ebs_config {
      size                 = "40"
      type                 = "gp3"
      volumes_per_instance = 1
    }
  }

  bootstrap_action {
    name = "Install Python Dependencies"
    path = "s3://${aws_s3_bucket.emr_bucket.id}/scripts/bootstrap.sh"
    args = []
  }
  log_uri = "s3://${aws_s3_bucket.emr_bucket.id}/logs/"

  step {
    name              = "Run Python Script"
    action_on_failure = "TERMINATE_CLUSTER"
    hadoop_jar_step {
      jar  = "command-runner.jar"
      args = concat([
      "spark-submit",
      "s3://${aws_s3_bucket.emr_bucket.id}/scripts/pyspark_queries.py",
      "--folder", "${var.data_folder}",
      "--scale_factor", tostring(var.scale_factor),
      "--queries"
      ], [for q in var.queries : tostring(q)])
    }
  }
  auto_termination_policy {
    idle_timeout = 60
  }

  service_role = aws_iam_role.emr_service_role.arn
  tags = {
    for-use-with-amazon-emr-managed-policies = "true"
  }
}

resource "aws_iam_role" "emr_service_role" {
  name = "EMR_Service_Role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "elasticmapreduce.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "emr_service_role_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEMRServicePolicy_v2"
}
resource "aws_iam_role_policy" "emr_pass_intsance_role_policy" {
  name   = "EMR_Pass_Instance_Role_Policy"
  role   = aws_iam_role.emr_service_role.name
  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
	  "Sid": "PassRoleForEC2",
	  "Effect": "Allow",
	  "Action": "iam:PassRole",
	  "Resource": "${aws_iam_role.emr_instance_role.arn}"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy" "emr_instance_profile_policy" {
  name = "EMR_S3_Access_Policy"
  role = aws_iam_role.emr_instance_role.name

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
	  "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${aws_s3_bucket.emr_bucket.id}/*",
        "arn:aws:s3:::${aws_s3_bucket.emr_bucket.id}"
      ]
    },
    {
      "Sid": "S3AccessToBodoExampleData",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${local.tpch_bucket}",
        "arn:aws:s3:::${local.tpch_bucket}/*"
      ]
    }
  ]
}
EOF
}
resource "aws_iam_role" "emr_instance_role" {
  name               = "EMR_EC2_Instance_Role"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_instance_profile" "emr_profile" {
  name = "EMR_EC2_Instance_Profile"
  role = aws_iam_role.emr_instance_role.name
}

resource "aws_subnet" "emr_subnet" {
  vpc_id            = aws_vpc.emr_vpc.id
  cidr_block        = "10.0.0.0/16"
  availability_zone = "us-east-2b"
  tags = {
    for-use-with-amazon-emr-managed-policies = "true"
  }
}

resource "aws_vpc" "emr_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = {
    for-use-with-amazon-emr-managed-policies = "true"
  }
}

resource "aws_security_group" "allow_access" {
  name                   = "allow_access"
  description            = "Allow inbound traffic"
  vpc_id                 = aws_vpc.emr_vpc.id
  revoke_rules_on_delete = true

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [aws_vpc.emr_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  depends_on = [aws_subnet.emr_subnet]

  lifecycle {
    ignore_changes = [
      ingress,
      egress,
    ]
  }

  tags = {
    for-use-with-amazon-emr-managed-policies = "true"
  }
}

resource "aws_internet_gateway" "emr_igw" {
  vpc_id = aws_vpc.emr_vpc.id
}

resource "aws_route_table" "emr_route_table" {
  vpc_id = aws_vpc.emr_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.emr_igw.id
  }
}

resource "aws_route_table_association" "emr_route_table_association" {
  subnet_id      = aws_subnet.emr_subnet.id
  route_table_id = aws_route_table.emr_route_table.id
}

data "aws_region" "current" {}

output "emr_cluster_id" {
  value = aws_emr_cluster.emr_cluster.id
}
output "emr_cluster_region" {
  value = data.aws_region.current.name
}
output "s3_bucket_id" {
  value = aws_s3_bucket.emr_bucket.id
}
