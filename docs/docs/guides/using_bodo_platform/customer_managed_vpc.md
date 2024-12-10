# Configuring Customer Managed VPC {#customer_managed_vpc}

## Overview

In a Bodo Managed VPC, Bodo creates and manages the VPC in your AWS account.
You can optionally create your Bodo workspaces in your own VPC, a feature known as customer-managed VPC.
You can use a customer-managed VPC to have more control over your network configurations to comply with specific cloud security and governance standards your organization may require.
Your workspace must use a customer-managed VPC to configure it to use AWS PrivateLink for any type of connection.

A customer-managed VPC is good solution if you have:

- Security policies that prevent PaaS providers from creating VPCs in your own AWS account.

- An approval process to create a new VPC, in which the VPC is configured and secured in a well-documented way by internal information security or cloud engineering teams.

### Creating a Workspace with Customer Managed VPC

To create a workspace, go to the "Workspaces" section in the sidebar and click "Create Workspace."

!!! info "Important"

```
Make sure you have a cloud configuration created before creating a workspace.
Refer to the [Setting AWS Credentials][setting_aws_credentials].
```

In the creation form:

1. Enter the **name** of the workspace

1. Select the **Cloud Configuration** from the dropdown

1. Select **region** to deploy the resources

1. In the Optional section, Select the **Network Configuration** from the dropdown

1. Click on "Create Workspace" to create the workspace.

!!! info "Important"

```
  Please refer to the following link for creating a network configuration: [Creating Network Configuration][network_configuration]
```
