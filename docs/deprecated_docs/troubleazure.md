# Troubleshooting Managed Bodo Cloud Platform Issues on Azure {#troubleshootingazure}

Here are solutions to potential issues you may encounter while using the
Bodo Cloud Platform.

## Cluster Creation Fails {#creationfail}

Most of cluster creation failures are usually due to one of the
following:

-   Your account hits Azure resource limits such as limits on the number
    of VNets and virtual machines
-   Your Azure credentials do not have the required permissions (see
    [how to set azure credentials][setting_azure_credentials])
-   Azure does not have enough of the requested resources (such as some
    of the large virtual machine sizes)

In case of failure, the logs are made available on the platform and
should provide some details regarding why the failure occurred. Even
though cluster creation was not successful, some Azure resources may
still have been provisioned. Click on the delete icon to remove all the
created resources, otherwise you may incur charges for the provisioned
Azure resources. You can try to create a cluster again after addressing
the underlying issue such as increasing limits or providing Azure
credentials with the required permissions.

## Cluster Deletion Fails {#deletionfail}

Failures during cluster deletion are very rare and usually only occur
when the provisioned resources have been manually modified in some way.
In these cases, logs are provided to help you diagnose the issue. For
instance, if logs indicate that some resource cannot be deleted due to a
dependent resource, you can try to delete the resource manually through
the [Azure Portal](https://portal.azure.com){target="blank"} and try to remove the
cluster through the platform again. The resources provisioned for the
cluster are tagged with the Cluster-ID in the resource group, making
them easy to identify and remove.

## Cleanup Shared Resources Manually {#manualcleanup}

As described in [Azure account cleanup][azure_account_cleanup],
an option to remove organization level shared resources provisioned by
Bodo in your Azure environment is provided. If you need to remove
resources manually (e.g. the process fails), you can simply remove all
the resources in the designated resource group whose name contains
*bodo*.

!!! note 
    Please ensure that you have removed all clusters and related
    resources before proceeding. Deleting the resources listed below may
    result in the platform losing access to those clusters for removal in
    the future.

The resources should be easy to identify within their respective
sections on the [Azure Portal](https://portal.azure.com) since their
names all contain `bodo`}. See [here][resources_created_in_azure_env] for a
list of resources the platform creates.

1.  Navigate to the [Azure Portal](https://portal.azure.com). Sign in if
    you are not already signed in. Navigate to the resource group that
    you're using for your Bodo resources (you can find the name on the
    *Settings* page). If all cluster specific resources have
    been deleted properly, the resource group should look something
    like:

![Azure Resource Group Cleanup](../platform_onboarding_screenshots/az-rg-cleanup.png#center)

2.  Look for all resources with *bodo* in their name, and delete them.

3.  Next, you will need to purge the key vault manually. Follow the
    instructions [here][manually_purge_azure_kayvault].

The steps above should remove the organization level resources
provisioned by Bodo in your Azure environment.

