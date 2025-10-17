# Bodo Managed Cloud Platform on Azure {#bodo_platform_azure}

## Setting Azure Credentials {#setting_azure_credentials}

To use Bodo on Azure, you need to link your Azure account to the Bodo
platform. This can be done using the *Cloud Configuration* page in the left
bar as shown in the picture below:

![Dashboard](../../platform_onboarding_screenshots/dashboard-az.gif#center)

In order to use the Bodo Platform to launch clusters and notebooks, you
must grant it permission to access your Azure account and provision the
required resources in it. You can do this by creating a [Service
Principal](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals)
for the Bodo Platform application and assigning a role to it.

### Create a Service Principal {#create_service_principal}

Login to your Azure Portal. Click on the icon next to the search bar to
open a *Cloud-Shell*. Execute the following command to create a service
principal:

```shell
az ad sp create --id APP_ID
```

where `APP_ID` is the Application ID for Bodo-Platform which
is displayed on the *Cloud Configuration Form*.

![Create SP on Azure Portal](../../platform_onboarding_screenshots/az-portal-create-sp.png#center)

Once you have created a service principal, you need to assign a role to
it. As shown below, go to the IAM section of your resource group and add a
`Contributor` Role and `Storage Blob Data Contributor` Role to the service principal you created for the Bodo Platform Application.

![Assign SP a Role](../../platform_onboarding_screenshots/az-assign-sp-role.gif#center)

!!! seealso "See Also"
    [Required Azure resource providers][required_az_resource_providers]


Once you have created the service principal and assigned a role to it,
you are now ready to fill the *Cloud Configuration* Form on the Bodo Platform.

![Cloud Configuration Page Azure](../../platform_onboarding_screenshots/cloud-configuration-az.png#center)

1.  Enter your Azure subscription ID in the **Subscription ID** field.
    You can find this in the *Subscription Overview*.

      ![Azure subscription ID](../../platform_onboarding_screenshots/az-subscription-id.png#center)

2.  Enter your Azure Tenant ID in the **Tenant ID** field. You can find
    this in *Azure AD*.

    ![Azure Tenant ID](../../platform_onboarding_screenshots/az-tenant-id.png#center)

3. Enter the name of the resource group where the infrastructure should be deployed.

4.  Select a **region** from the dropdown list. This region refers to the region of
    the resource group mentioned in the previous step. We will also create a storage account and a blob container in this region to store metadata such as the state of the deployed infrastructure, logs, etc.


5.  Click on `CREATE`.

!!! note
    We highly recommend that you ensure sufficient limits on
    your Azure subscription to launch resources. See
    [here][resources_created_in_azure_env] for the
    resources required for Bodo Cloud Platform.

## Required Resource Providers on Azure subscription {#required_az_resource_providers}

Ensure that the following resource providers are registered on your
Azure subscription:

-   Microsoft.Authorization
-   Microsoft.Compute
-   Microsoft.KeyVault
-   Microsoft.ManagedIdentity
-   Microsoft.Network
-   Microsoft.Resources
-   Microsoft.Storage

![Azure Required Resource Providers](../../platform_onboarding_screenshots/az-resource-providers.png#center)



## Resources Created in Your Azure Environment {#resources_created_in_azure_env}

Bodo deploys cluster/notebook resources in your own Azure environment to
ensure security of your data. Below is a list of Azure resources that
the Bodo Platform creates in your account to enable clusters and
notebooks.

  Azure Service                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Purpose
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------
  [Virtual Machines](https://azure.microsoft.com/en-us/services/virtual-machines/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Cluster/notebook workers
  [Storage Accounts](https://azure.microsoft.com/en-us/product-categories/storage/), [File-Shares](https://azure.microsoft.com/en-us/services/storage/files/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Shared file system for clusters
  [Virtual Network with Subnets and NAT Gateway](https://azure.microsoft.com/en-us/services/virtual-network/), [Public IP](https://docs.microsoft.com/en-us/azure/virtual-network/associate-public-ip-address-vm), [NIC](https://docs.microsoft.com/en-us/azure/virtual-network/virtual-network-network-interface-vm), [Proximity Placement Groups](https://docs.microsoft.com/en-us/azure/virtual-machines/co-location), [Availability Sets](https://docs.microsoft.com/en-us/azure/virtual-machines/availability-set-overview), [Security Groups](https://docs.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview), ...   |Secure networking for clusters/notebooks
  [Blob Containers](https://azure.microsoft.com/en-us/services/storage/blobs/),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Resource states
  [KeyVault](https://azure.microsoft.com/en-us/services/key-vault/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Cluster secrets (e.g. SSH keys)
  [VM Identity](https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm) for Clusters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Allow cluster workers to access resources above

!!! note
    These resources incur additional Azure infrastructure charges and are
    not included in the Bodo Platform charges.

## Using Bodo Platform

Please refer to the [platform usage guides][bodo-cloud-platform-guides] to explain the basics of using the Bodo Cloud Platform and associated concepts.


