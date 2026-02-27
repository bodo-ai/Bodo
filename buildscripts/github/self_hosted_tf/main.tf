locals {
  prefix     = "bodo-gh-ci"
  aws_region = "us-east-2"
  version    = "6.6.0"
}


resource "random_password" "random" {
  length = 20
}

resource "aws_resourcegroups_group" "resourcegroups_group" {
  name = "${local.prefix}-group"
  resource_query {
    query = templatefile("${path.module}/templates/resource-group.json", {
      example = local.prefix
    })
  }
}

module "runners" {
  source = "github-aws-runners/github-runner/aws//modules/multi-runner"
  # Same as local.version
  version = "6.6.0"

  # Multi-Size Runners to Use
  # Assume all Runners are Using Amazon Linux 2023
  # Names & Sizes are Based on CodeBuild Instance Types
  multi_runner_config = {
    "small" = {
      matcherConfig : {
        labelMatchers = [["self-hosted", "small"], ["self-hosted-small"]]
        exactMatch    = true
      }

      # Recommended disabled for ephemeral runners
      fifo = false

      runner_config = merge(local.base_runner_config, {
        # Instance Type(s) (Multiple Options to Choose for Spot)
        instance_types = ["c7i.large", "m7i.large", "r7i.large", "c7a.large", "m7a.large", "r7a.large", "c6i.large", "c6a.large", "c6id.large", "m6i.large", "m6a.large", "m6id.large", "r6i.large", "r6a.large", "r6in.large", "r6id.large", "i4i.large"]
        # Prefix runners with the environment name
        runner_name_prefix = "${local.prefix}_small_"
        # Max # of Runners of this Size
        runners_maximum_count = 20
      })
    }

    "medium" = {
      matcherConfig : {
        labelMatchers = [["self-hosted", "medium"], ["self-hosted-medium"]]
        exactMatch    = true
      }

      # Recommended disabled for ephemeral runners
      fifo = false

      runner_config = merge(local.base_runner_config, {
        # Instance Type(s) (Multiple Options to Choose for Spot)
        instance_types = [
          // Disable for now until OOM is confirmed
          // "c7i.xlarge", "c7i-flex.xlarge", "c6i.xlarge", "c6a.xlarge", "c6id.xlarge", "c6in.xlarge",
          "m7i.xlarge", "r7i.xlarge",
          "m7i-flex.xlarge",
          "m6i.xlarge", "r6i.xlarge",
          "m6a.xlarge", "r6a.xlarge",
          "m6id.xlarge", "r6id.xlarge",
          "m6in.xlarge", "r6in.xlarge",
          "i4i.xlarge"
        ]
        # Prefix runners with the environment name
        runner_name_prefix = "${local.prefix}_medium_"
        # Max # of Runners of this Size
        runners_maximum_count = 100
      })
    }

    "large" = {
      matcherConfig : {
        labelMatchers = [["self-hosted", "large"], ["self-hosted-large"]]
        exactMatch    = true
      }

      # Recommended disabled for ephemeral runners
      fifo = false

      runner_config = merge(local.base_runner_config, {
        # Instance Type(s) (Multiple Options to Choose for Spot)
        instance_types = ["c7i.2xlarge", "c6i.2xlarge", "t3.2xlarge", "m6i.2xlarge", "m6id.2xlarge", "c5.2xlarge"]
        # Prefix runners with the environment name
        runner_name_prefix = "${local.prefix}_large_"
        # Max # of Runners of this Size
        runners_maximum_count = 10
      })
    }

    "xlarge" = {
      matcherConfig : {
        labelMatchers = [["self-hosted", "xlarge"], ["self-hosted-xlarge"]]
        exactMatch    = true
      }

      # Recommended disabled for ephemeral runners
      fifo = false

      runner_config = merge(local.base_runner_config, {
        # Instance Type(s) (Multiple Options to Choose for Spot)
        instance_types = ["c5.18xlarge", "c5n.18xlarge", "c5d.18xlarge"]
        # Prefix runners with the environment name
        runner_name_prefix = "${local.prefix}_xlarge_"
        # Max # of Runners of this Size
        runners_maximum_count = 5
      })
    }

    "multi-gpu" = {
      matcherConfig : {
        labelMatchers = [["self-hosted", "multi-gpu"], ["self-hosted-multi-gpu"]]
        exactMatch    = true
      }

      # Recommended disabled for ephemeral runners
      fifo = false

      runner_config = merge(local.base_gpu_runner_config, {
        # Instance Type(s) (Multiple Options to Choose for Spot)
        instance_types = ["g4dn.12xlarge", "g5.12xlarge", "g6.12xlarge"]
        # Prefix runners with the environment name
        runner_name_prefix = "${local.prefix}_multi_gpu_"
      })
    }
  }

  # General AWS Properties
  aws_region = local.aws_region
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  prefix     = local.prefix
  tags = {
    Project = "BodoGHActionsCI"
  }

  # General GitHub Properties
  github_app = {
    id             = var.github_app_id
    key_base64     = var.github_key_base64
    webhook_secret = random_password.random.result
  }

  # Zip Files for Lambdas
  webhook_lambda_zip                = "webhook-${local.version}.zip"
  runner_binaries_syncer_lambda_zip = "runner-binaries-syncer-${local.version}.zip"
  runners_lambda_zip                = "runners-${local.version}.zip"

  # Termination Watcher Config
  instance_termination_watcher = {
    enable = true
    zip    = "termination-watcher-${local.version}.zip"
  }

  # Additional Features
  # Enable debug logging for the lambda functions
  # log_level = "debug"
}

module "webhook_github_app" {
  source     = "github-aws-runners/github-runner/aws//modules/webhook-github-app"
  version    = "6.6.0"
  depends_on = [module.runners]

  github_app = {
    key_base64     = var.github_key_base64
    id             = var.github_app_id
    webhook_secret = random_password.random.result
  }
  webhook_endpoint = module.runners.webhook.endpoint
}

module "spot_termination_watchter" {
  source     = "github-aws-runners/github-runner/aws//modules/termination-watcher"
  version    = "6.6.0"
  depends_on = [module.runners]

  config = {
    prefix = "global"
    zip    = "termination-watcher-${local.version}.zip"

    tag_filters = {
      "ghr:Application" = "github-action-runner"
    }

    metrics = {
      enable = true
      metric = {
        enable_spot_termination_warning = true
      }
    }
  }
}
