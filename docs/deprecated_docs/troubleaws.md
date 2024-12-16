# Troubleshooting Managed Bodo Cloud Platform Issues on AWS {#troubleshootingaws}

Here are solutions to potential issues you may encounter while using the
Bodo Cloud Platform. 

## Cluster Creation Fails {#creationfail}

Most of cluster creation failures are usually due to one of the
following:

-   Your account hits AWS resource limits such as limits on the number
    of VPCs and EC2 instances
-   Your AWS credentials do not have the required permissions (see
    [how to set aws credentials][setting_aws_credentials])
-   AWS does not have enough of the requested resources (such as some of
    the large EC2 instances)

In case of failure, the logs are made available on the platform and
should provide some details regarding why the failure occurred. Even
though cluster creation was not successful, some AWS resources may still
have been provisioned. Click on the delete icon to remove all the
created resources, otherwise you may incur charges for the provisioned
AWS resources. You can try to create a cluster again after addressing
the underlying issue such as increasing limits or providing AWS
credentials with the required permissions.

## Cluster Deletion Fails {#deletionfail}

Failures during cluster deletion are very rare and usually only occur
when the provisioned resources have been manually modified in some way.
In these cases, logs are provided to help you diagnose the issue. For
instance, if logs indicate that some resource cannot be deleted due to a
dependent resource, you can try to delete the resource manually through
the [AWS Management Console](https://aws.amazon.com/console/){target="blank"}
and try to remove the cluster through the platform again.

## Cleanup Shared Resources Manually {#manualcleanup}

As described in the [AWS account cleanup section][aws_account_cleanup], an
option to remove organization level shared resources provisioned by Bodo
in your AWS environment is provided. If you need to remove resources
manually (e.g. the process fails), below is the list of organization
level resources and the order to remove them.

!!! note 
    Please ensure that you have removed all clusters and related
    resources before proceeding. Deleting the resources listed below may
    result in the platform losing access to those clusters for removal in
    the future.

The resources should be easy to identify within their respective
sections on the [AWS Management Console](https://aws.amazon.com/console/){target="blank"} 
since their names are all prefixed with `bodo`.

1.  Navigate to the [AWS Management Console](https://aws.amazon.com/console/){target="blank"}. 
    Sign in if you are not already signed in. Make sure you have selected the region from which
    you want to remove the shared resources.

2.  Click on *Services* in the top-right corner. Navigate to
    the *EC2* section (under *Compute*) and then
    to *Network Interfaces* in the sidebar (under *Network &
    Security*). You will see two Network Interfaces. One of
    them is required for an EFS Mount (shared storage), and the other is
    required by a NAT Gateway. These dependent resources need to be
    removed first.

    a.  Click on *Services* and navigate to the
        *EFS* section (under *Storage*). Click
        on *File Systems* in the sidebar. Delete the File
        System prefixed with `bodo` by selecting it and
        clicking on *Delete*.
    b.  Click on *Services* and navigate to the
        *VPC* section (under *Networking & Content
        Delivery*). Select *NAT Gateways* in the
        sidebar (under *Virtual Private Cloud*). Select the
        NAT Gateway prefixed with *bodo* and delete it.

    Navigate back to *Network Interfaces* in the
    *EC2* section and ensure that the two ENIs are deleted
    (or have the status *available*). This may take a few
    minutes in some cases.

3.  Click on *Services* and navigate to the
    *VPC* section (under *Networking & Content
    Delivery*). Select *Your VPCs* in the
    sidebar (under *Virtual Private Cloud*). Select the VPC
    prefixed with *bodo* and delete it. If there is a
    dependency warning, wait for a few minutes and try again. You can
    also try to delete the linked dependent resources manually if it
    does not resolve on its own.

4.  Click on *Services* in the top-right corner. Navigate to
    the *EC2* section (under *Compute*) and
    select *Elastic IPs* in the sidebar (under *Network &
    Security*). Select the EIP prefixed with
    *bodo* and select *Release Elastic IP
    addresses* under *Actions*.

5.  Click on *Services* in the top-right corner. Navigate to
    the *Key Management Service (KMS)* section (under
    *Security, Identity, & Compliance*) and select *Customer
    managed keys* in the sidebar. Click on the key prefixed
    with *bodoai-kms*. Go to the *Aliases* tab.
    There should be a single alias defined. Select this alias and delete
    it. Next, click on *Key actions* (top-right) and select
    *Schedule key deletion*.

    !!! info "Optional"
        Reduce the *Waiting period* from 30 days to 7 days.

    Next, check *Confirm that you want to delete this key in XX
    days* and click on *Schedule deletion*.

6.  Finally, click on *Services* in the top-right corner and
    navigate to *Systems Manager* (under *Management &
    Governance*). Select *Parameter Store* from
    sidebar. Look for parameters prefixed with `/<EXTERNAL_ID>`, where
    `EXTERNAL_ID` is the same as the External ID visible on the Settings
    page on the Bodo Platform (see
    [how to create an iam role manually][create_iam_role_manually]). Select
    all these parameter entries and delete them.

The steps above should remove the organization level resources
provisioned by Bodo in your AWS environment.

