# Bodo Managed Cloud Platform on Azure {#bodo_platform_azure}

## Registration

a.  [Contact Bodo](https://bodo.ai/contact) to be onboarded onto Bodo
    Cloud Platform on Azure. You will be provided with an onboarding
    link.

b.  The provided link will take you to Bodo Platform's registration
    page.

c.  Fill out the fields with your information. If this is your
    individual account, use a unique name such as
    *firstname_lastname* for the **Organization Name**
    field.

d.  Check the box for accepting terms and conditions and click on
    `SIGN UP`:
    ![Signup-Page](../platform_onboarding_screenshots/signup.png#center)

e.  A page confirming that an activation link was sent to your email
    will appear. Please open the email and click on the activation link:
    ![Signup-Page-Confirmation](../platform_onboarding_screenshots/signup-conf.png#center)
    Clicking on the confirmation link will take you to the bodo platform
    page where you can use your newly created credentials to sign in:
    ![Login-Page](../platform_onboarding_screenshots/login.png#center)

## Setting Azure Credentials {#setting_azure_credentials}

To use Bodo on Azure, you need to link your Azure account to the Bodo
platform. This can be done either using the *Settings* page in the left
bar or the first item in the *Onboarding* list highlighted in green as
shown in the picture below:

![Dashboard](../platform_onboarding_screenshots/dashboard-az.png#center)

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
is displayed on the *Settings* Page.

![Create SP on Azure Portal](../platform_onboarding_screenshots/az-portal-create-sp.png#center)

Once you have created a service principal, you need to assign a role to
it. You can assign a role to this service principal at either a
subscription level or a resource group level. Subscription level
permissions are only required if you want Bodo to create a new resource
group. If you provide an existing resource group, only permissions at
the resource group level are required. As shown below, go to the IAM
section of your subscription or resource group and add a
`Contributor` Role to the service principal you created for
the Bodo Platform Application.

![Assign SP a Role](../platform_onboarding_screenshots/az-assign-sp-role.png#center)

!!! seealso "See Also"
    [Required Azure resource providers][required_az_resource_providers]


Once you have created the service principal and assigned a role to it,
you are now ready to fill the *Settings* Form on the Bodo Platform.

1.  Enter your Azure subscription ID in the **Subscription ID** field.
    You can find this in the *Subscription Overview*.

      ![Azure subscription ID](../platform_onboarding_screenshots/az-subscription-id.png#center)

2.  Enter your Azure Tenant ID in the **Tenant ID** field. You can find
    this in *Azure AD*.

    ![Azure Tenant ID](../platform_onboarding_screenshots/az-tenant-id.png#center)

3.  If you've given Bodo subscription level permissions and want Bodo
    to create a new resource group in your Azure subscription, enter the
    name of the resource group you want it to create in the **Resource
    Group** field. A suggested name is pre-filled for you. If you've
    given Bodo resource group level permissions to an existing resource
    group, enter the name of this resource group.

4.  Select a **region** from the dropdown list. This is the region that
    all Bodo resources will be deployed in. If you're providing an
    existing resource group, this must be the region this resource group
    is located in.

5.  Click on `SAVE`.

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

![Azure Required Resource Providers](../platform_onboarding_screenshots/az-resource-providers.png#center)

!!! seealso "See Also"
    [Bodo Cloud Platform][bodo_platform]


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


## Azure Account Cleanup {#azure_account_cleanup}

As explained in [earlier][resources_created_in_azure_env], the platform creates two types of resources in the users'
Azure environments: organization level resources and cluster specific
resources. Organization level resources are created by the platform to
set up shared resources (such as a VNets, File-Share, etc) that are used
later by all created resources. Cluster specific resources (such as
virtual machines, NICs, etc) are created by the platform to host/manage
a specific cluster. This includes notebooks and corresponding resources
as well. The cluster specific resources are removed when you request a
cluster to be removed. The organization level resources persist in the
user account so they can be used by clusters deployed in the future.
However, if you need to remove these resources for any reason (Azure
resource limits, etc.), an option to do so is provided. Navigate to the
*Settings* page and click on `Show Advanced` in the bottom-right corner.

![Settings-Account-Cleanup](../platform_onboarding_screenshots/settings-az-ac-cleanup.png#center)

This will bring up a section called *Azure Resource Cleanup*.

![Advanced-Settings-Account-Cleanup](../platform_onboarding_screenshots/settings-adv-az-ac-cleanup.png#center)

Select the region from which you would like to remove these resources
(i.e. the region in which the resources you want to delete have been
created), and click `CLEANUP AZURE RESOURCES`. Note that this will only
work if you don't have any active clusters in that region deployed
through the platform. Else, the request will be rejected, and you'll be
asked to remove all clusters in that region before trying again.
Removing active clusters (including clusters with a *FAILED* status) is
necessary because this process will make them inaccessible to the
platform.

![Advanced-Settings-Account-Cleanup Completion](../platform_onboarding_screenshots/az-acc-cleanup-completion.png#center)

The KeyVault deleted as part of this process needs to be purged manually
through the [Azure Portal](https://portal.azure.com) if you plan to
create resources on the platform again. See how to [manually purge Azure KeyVault][manually_purge_azure_kayvault].

### Manually Purge Azure Keyvault {#manually_purge_azure_kayvault}

Purging key vaults requires subscription level permissions. You can read
more about this
[here](https://docs.microsoft.com/en-us/azure/key-vault/general/soft-delete-overview)
and
[here](https://docs.microsoft.com/en-us/azure/key-vault/general/key-vault-recovery).
To avoid having to assign subscription level roles to Bodo Platform's
service principal, we require users to do this step manually.

1.  Navigate to *Key vaults* on your [Azure
    Portal](https://portal.azure.com).

      ![Azure Portal KVs](../platform_onboarding_screenshots/az-manual-kv-purge-portal.png#center)

2.  Click on `Manage deleted vaults`. In the form, select the
    subscription associated with Bodo KeyVault to see a list of deleted
    key vaults.

    ![Azure Portal KVs List](../platform_onboarding_screenshots/az-manual-kv-purge-kvs.png#center)

3.  Select the key vault with `bodo` in its name, click on
    `Purge` and confirm by clicking `Delete`.

    ![Azure Portal KV Purge Step](../platform_onboarding_screenshots/az-manual-kv-purge-step.png#center)

4.  As shown in the notification, the purge process can take up to 10
    minutes to complete. The purged key vault may continue to show up on
    the list of deleted key vaults until it has been successfully
    purged.

    ![Azure Portal KV Purge Notification](../platform_onboarding_screenshots/az-manual-kv-purge-notification.png#center)

5.  Once the key vault has been successfully purged, the list of deleted
    keyvaults should not feature it. At this point you can use the Bodo
    Platform again to provision clusters, etc.

    ![Azure Portal KV Purge Completion](../platform_onboarding_screenshots/az-manual-kv-purge-completion.png#center)

!!! seealso "See Also"
    [Troubleshooting Managed Bodo Cloud Platform Issues on Azure][troubleshootingazure]

