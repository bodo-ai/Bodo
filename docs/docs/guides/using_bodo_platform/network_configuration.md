# Creating Network Configuration {#network_configuration}

# Overview
In a Bodo Managed Workspace, Bodo creates and manages the VPC in your AWS account.
You can optionally create your Bodo workspaces in your own VPC, a feature known as customer-managed VPC.
You can use a customer-managed VPC to have more control over your network configurations to comply with specific cloud security and governance standards your organization may require.
In order to configure your workspace to use AWS PrivateLink for any type of connection, it must use a customer managed VPC. 

To create a Customer Managed VPC or Configure PrivateLink, you must create a network configuration in the Bodo Platform.

# Network Configuration Requirements

## VPC Requirements
A Customer Managed VPC must meet the following requirements to be able to deploy Bodo workspaces

### VPC Region
Currently, Bodo supports **us-east-1, us-east-2, us-west-1, us-west-2 and eu-west-1** regions for customer-managed VPCs.

### VPC sizing
A single VPC can be shared with multiple workspaces in the same AWS account.

However, Bodo recommends having unique **subnets** and **security groups** for each workspace to avoid any potential conflicts.
Make sure to size your VPC and subnets accordingly. Bodo assigns a single IP address for each node.
The total number of instances for each subnet should not exceed the number of available IP addresses in the subnet.

### VPC IP address Ranges
Bodo doesn't limit netmasks for the workspace VPC but each workspace subnet must have netmask between `/17` and `/26`

!!! info "Important"

    If you have configured secondary CIDR blocks for your VPC, 
    make sure that the subnets for the Bodo workspace are configured with the same VPC CIDR block.

### DNS
The VPC must have DNS resolution and DNS hostnames enabled.

### Subnets
Bodo must have access to at least two subnets for each workspace, with each subnet in a different Availability Zone (AZ).
Your network configuration could have more than one subnet per AZ, but Bodo requires at least two subnets in different AZs.

#### Subnet Requirements
* Each subnet must have a netmask between `/17` and `/26`.

* Subnets must be private

* Subnets must have a route to the internet gateway or NAT gateway for internet access.

* The NAT gateway must be in separate subnet called Public Subnet that routes `0.0.0.0/0` traffic to an internet gateway.

* The route table for the workspace subnets must have `0.0.0.0/0` traffic that targets the appropriate network device.

* `0.0.0.0/0` traffic must be routed to the NAT gateway or your own managed NAT device or proxy appliance.

!!! info "Important"

    When using PrivateLink, you don't need to have a route to the internet gateway or NAT gateway for internet access.


### Security Groups
A Bodo workspace must have access to at least one AWS security group and no more than 5 security groups.
You can reuse the same security group for multiple workspaces, but it is recommended to have a separate security group for each workspace.

Security groups must have the following rules:

**Egress(Outbound):**

* Allow all TCP access to the workspace security group (for internal traffic)

* Allow TCP access to `0.0.0.0/0` for these ports

    * **443**: For Bodo Infrastructure access, cloud data sources etc.

    * **2049**: For EFS access to the shared file system

    * 80(optional): Needed for [Snowflake OCSP checks ](https://docs.snowflake.com/en/user-guide/ocsp#ca-site-and-ocsp-responder-hosts-used-by-snowflake)(Required for Snowflake customers)


**Ingress(Inbound):**

* Allow all TCP access to the workspace security group

!!! info "Important"

    Workspaces must have outbound access from the VPC to the public network.


### Subnet Network ACLs
Subnet Network ACLs must not deny any traffic ingress or egress the workspace security group.
If defined, the ACLs must have the following rules:


**Egress(Outbound):**

* Allow all TCP access to the workspace security group (for internal traffic)

* Allow TCP access to 0.0.0.0/0 for these ports

    * **443**: For Bodo Infrastructure access, cloud data sources etc.

    * **2049**: For EFS access to the shared file system

    * 80(optional): Needed for [Snowflake OCSP checks ](https://docs.snowflake.com/en/user-guide/ocsp#ca-site-and-ocsp-responder-hosts-used-by-snowflake)(Required for Snowflake customers)


**Ingress(Inbound):**

* Allow all TCP access to the workspace security group


# Creating Network Configuration

To create a network configuration, go to the "Configurations" section in the sidebar and click on "Network Configurations" tab 
and click on "Create Network Configuration" on top-right corner.

In the network configuration form:

1. Enter the **name** of the network configuration
2. Select the **region**
3. Enter the **VPC ID**
4. Enter the **Subnet IDs** (Minimum 2)
5. Enter the **Security Group IDs** (Minimum 1)
6. Select **Enabled AWS PrivateLink** if you want to enable PrivateLink for the workspace
7. Enter the **VPC endpoint IDs** for the services you want to connect to using PrivateLink (S3, SSM, Bodo Cluster endpoints are mandatory for PrivateLink)
8. Click on **Create Network Configuration** to create the network configuration.

    ![Network Configuration Form](../../platform_onboarding_screenshots/network-config-form.png#center)