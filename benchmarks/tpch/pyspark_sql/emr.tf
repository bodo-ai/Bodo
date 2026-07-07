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
  key    = "scripts/sql_on_spark_queries.py"
  source = "./sql_on_spark_queries.py"
}

resource "aws_s3_object" "bootstrap_script" {
  bucket  = aws_s3_bucket.emr_bucket.id
  key     = "scripts/bootstrap.sh"
  content = <<EOF
#!/bin/bash
sudo pip install -U pandas numpy==1.26.4 pyarrow

# create app layout and download files from S3
APP_DIR=/home/hadoop/app
mkdir -p $${APP_DIR}/scripts $${APP_DIR}/sql
aws s3 cp s3://${aws_s3_bucket.emr_bucket.id}/scripts/ $${APP_DIR}/scripts/ --recursive
aws s3 cp s3://${aws_s3_bucket.emr_bucket.id}/sql/     $${APP_DIR}/sql/     --recursive

# ensure ownership
chown -R hadoop:hadoop $${APP_DIR}
EOF
}

resource "aws_s3_object" "sql_script_1" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q01.sql"
  source = "../sql/q01.sql"
}

resource "aws_s3_object" "sql_script_2" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q02.sql"
  source = "../sql/q02.sql"
}

resource "aws_s3_object" "sql_script_3" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q03.sql"
  source = "../sql/q03.sql"
}

resource "aws_s3_object" "sql_script_4" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q04.sql"
  source = "../sql/q04.sql"
}

resource "aws_s3_object" "sql_script_5" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q05.sql"
  source = "../sql/q05.sql"
}

resource "aws_s3_object" "sql_script_6" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q06.sql"
  source = "../sql/q06.sql"
}

resource "aws_s3_object" "sql_script_7" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q07.sql"
  source = "../sql/q07.sql"
}

resource "aws_s3_object" "sql_script_8" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q08.sql"
  source = "../sql/q08.sql"
}

resource "aws_s3_object" "sql_script_9" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q09.sql"
  source = "../sql/q09.sql"
}

resource "aws_s3_object" "sql_script_10" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q10.sql"
  source = "../sql/q10.sql"
}

resource "aws_s3_object" "sql_script_11" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q11.sql"
  source = "../sql/q11.sql"
}

resource "aws_s3_object" "sql_script_12" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q12.sql"
  source = "../sql/q12.sql"
}

resource "aws_s3_object" "sql_script_13" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q13.sql"
  source = "../sql/q13.sql"
}

resource "aws_s3_object" "sql_script_14" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q14.sql"
  source = "../sql/q14.sql"
}

resource "aws_s3_object" "sql_script_15" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q15.sql"
  source = "../sql/q15.sql"
}

resource "aws_s3_object" "sql_script_16" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q16.sql"
  source = "../sql/q16.sql"
}

resource "aws_s3_object" "sql_script_17" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q17.sql"
  source = "../sql/q17.sql"
}

resource "aws_s3_object" "sql_script_18" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q18.sql"
  source = "../sql/q18.sql"
}

resource "aws_s3_object" "sql_script_19" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q19.sql"
  source = "../sql/q19.sql"
}

resource "aws_s3_object" "sql_script_20" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q20.sql"
  source = "../sql/q20.sql"
}

resource "aws_s3_object" "sql_script_21" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q21.sql"
  source = "../sql/q21.sql"
}

resource "aws_s3_object" "sql_script_22" {
  bucket = aws_s3_bucket.emr_bucket.id
  key    = "sql/q22.sql"
  source = "../sql/q22.sql"
}

resource "aws_emr_cluster" "emr_cluster" {
  name          = "EMR-Cluster"
  release_label = "emr-7.9.0"
  applications  = ["Hadoop", "Spark"]

  ec2_attributes {
    instance_profile                  = aws_iam_instance_profile.emr_profile.arn
    subnet_id                         = aws_subnet.emr_subnet.id
    emr_managed_master_security_group = aws_security_group.allow_access.id
    emr_managed_slave_security_group  = aws_security_group.allow_access.id
  }

  master_instance_group {
    instance_type = "c6i.xlarge"
  }

  core_instance_group {
    instance_type  = "r6i.16xlarge"
    instance_count = 4

    ebs_config {
      size                 = "300"
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

  dynamic "step" {
    for_each = var.queries

    content {
      name              = "Run TPCH Query ${step.value}"
      action_on_failure = "CONTINUE"

      hadoop_jar_step {
        jar = "command-runner.jar"

        args = [
          "spark-submit",
          "/home/hadoop/app/scripts/sql_on_spark_queries.py",
          "--folder", var.data_folder,
          "--scale_factor", tostring(var.scale_factor),
          "--queries", tostring(step.value)
        ]
      }
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
  name = "EMR_EC2_InstanceProfile"
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
