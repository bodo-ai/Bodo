# Organization Basics

This page describes the fundamental components of the Bodo Cloud Platform and how they are organized. The platform is designed to be a multi-tenant system, which has the entities described below.

### Organizations

Organizations on the Bodo Cloud Platform are tenants for billing and cloud resource management purposes.
An organization can have multiple [workspaces][workspaces] and [cloud configurations][cloud-configurations], and users can be part of multiple organizations.

![User-Mgmt](../../platform2-screenshots/user_mgmt.png#center)

### Cloud-Configurations

A *cloud-configuration* is an entity used to store information about your AWS or Azure account.
It consists of:

1. Details regarding the trust relationship between the platform and your cloud provider account.
   For AWS accounts, this is done through a cross-account IAM role.
   For Azure account, this is done through a service principal (scoped to a specific resource group)
   for the Bodo Platform application.
   This allows the platform to provision and manage cloud resources in your account.
2. Details regarding metadata storage. The platform needs to store
   specific metadata to carry out its functions, such as the state of your various cloud deployments, logs, etc.
   On AWS, this data is stored in an S3 bucket and a DynamoDB table.
   On Azure, this data is stored in a storage container.


![Cloud-Configs](../../platform2-screenshots/cloud_configs.png#center)

### Workspaces

A *workspace* on the Bodo Cloud Platform consists of:

1. A shared filesystem where you can collaborate with your team on your projects.
2. Networking infrastructure such as virtual networks, security groups, and subnets in which
   your compute clusters and Jupyter servers will be securely deployed.

A workspace is tied to a particular cloud configuration and has its own user management, i.e.,
you can have different subsets of users with different sets of roles
and permissions in different workspaces within the same organization.

!!! info "Important"
     If a user who is not part of the organization is invited to a workspace in the organization,
    they are automatically added to the organization with minimal permissions.

![Workspaces](../../platform2-screenshots/workspace.png#center)

To create a workspace, go to the "Workspaces" section in the sidebar and click "Create Workspace." In the
creation form, enter the name of the workspace, select the cloud configuration to use for provisioning
it and the region where it should be deployed, and click on "Create Workspace."

![Create-Workspace](../../platform2-screenshots/create-workspace-form.png#center)

This will start the workspace deployment. When the workspace is in the "READY" state, click on
the button next to it to enter it.

![Enter-Workspace](../../platform2-screenshots/enter-workspace.png#center)
