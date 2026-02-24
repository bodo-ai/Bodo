# Basic Config for Each Runner
# By default, all runners should be:
# - Amazon Linux 2023
# - x64
# - Spot with Fallback to On-Demand
# - Ephemeral (Recreated between jobs)
# - Accessible via SSM
# The main difference between runners is the instance type / size.

locals {
  base_runner_config = {
    # AMI
    runner_os           = "linux"
    runner_architecture = "x64"
    # Use Spot Instances
    instance_target_capacity_type = "spot"
    # How to Choose Between Spot Sizes
    # For now, choose the option most likely to not be interrupted
    instance_allocation_strategy = "capacity-optimized"
    # Use On-Demand Instances if Spot Instances are Unavailable
    enable_on_demand_failover_for_errors = ["InsufficientInstanceCapacity", "UnfulfillableCapacity"]

    # Let the module manage the service linked role
    create_service_linked_role_spot = true
    # Ephemeral Runners are not reused between jobs
    # Safer but slower option
    enable_ephemeral_runners = true
    # Enable Access to the Runners via SSM
    enable_ssm_on_runners = true
    # Check Every 30 Seconds to Scale Down
    # TODO: Default is 5 seconds. Too much overhead?
    scale_down_schedule_expression = "cron(*/30 * * * ? *)"
    # Organizational Runners Can be Used in Multiple Repos
    enable_organization_runners = true

    # Override Delay of Events in Seconds
    # Should be 0 for ephemeral runners
    delay_webhook_event = 0

    # Allow Runners to Assume the Following Roles
    runner_iam_role_managed_policy_arns = [
      "arn:aws:iam::427443013497:policy/AssumeEngineCIRole", # Assume the EngineCIRole
      "arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess" # DynamoDB Read Access for Credstash
    ]

    # Give the runners a 50 GB EBS Volume
    block_device_mappings = [{
      volume_size = 50
    }]
  }

  base_gpu_runner_config = merge(local.base_runner_config, {
    # Max # of Runners of this Size
    runners_maximum_count = 2
    # Use Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)
    ami = {
      id_ssm_parameter_arn = "arn:aws:ssm:${local.aws_region}::parameter/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
    }
    # GPU AMI requires >=75 GB of EBS Storage
    block_device_mappings = [{
      volume_size = 100
    }]
  })
}
