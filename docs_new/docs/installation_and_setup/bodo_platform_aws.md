# Bodo Managed Cloud Platform on AWS {#bodo_platform_aws}

## Registration

a.  Subscribe through the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/B08NY29SMQ){target="blank"}.

b.  After confirming your subscription, you'll be directed to Bodo
    Platform's registration page.

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

## Setting AWS Credentials {#setting_aws_credentials}

To use Bodo on AWS, you need to link your AWS account to the Bodo
platform. This can be done either using the *Settings* page in the left
bar or the first item in the *Onboarding* list highlighted in green as
shown in the picture below:

![Dashboard](../platform_onboarding_screenshots/dashboard.png#center)

To be able to use the Bodo Platform to launch clusters and notebooks,
you must grant it permission to access your AWS account and provision
the required resources in it. This can be done through an [AWS Cross
Account IAM
Role](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)
for the Bodo Platform.

### Create a Cross-Account IAM Role {#create_iam_role}

There are two ways to create such an IAM Role, (a) you can create it
manually, or (b) you can provide us with [Access
Keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html){target="blank"}
and we can create an IAM role in your AWS account. We provide directions
for both these methods below.

#### Create the IAM Role Manually {#create_iam_role_manually}
***Create the IAM Role Manually***

1.  Log in to the [AWS Management
    Console](https://aws.amazon.com/console/) and navigate to the
    **IAM** Service.

2.  Select the *Roles* tab in the sidebar, and click `Create Role`.

3.  In **Select type of trusted entity**, select `Another AWS Account`.

4.  Enter the Bodo Platform Account ID `481633624848` in the
    **Account ID** field.

5.  Check the `Require external ID` option.

    ![Create Role Form Step 1](../create_role_screenshots/create_iam_role_manual_form_step1.png#center)
    
    In the **External ID** field, copy over the External ID from the
    *Settings* page on the Bodo Platform.
    
    ![External ID Platform](../create_role_screenshots/create_iam_role_manual_externalId.png#center)

6.  Click the `Next: Permissions` button.

7.  Click the `Next: Tags` button.

8.  Click the `Next: Review` button.

9.  In the **Role name** field, enter a role name, e.g.
    `BodoPlatformUser`.

    ![Create Role Form Review](../create_role_screenshots/create_iam_role_manual_review_step.png#center)

10. Click `Create Role`. You will be taken back to the list of IAM Roles
    in your account.

11. In the list of IAM Roles, click on the role you just created.

12. Click on `Add inline policy`.

    ![Create Role Summary Page](../create_role_screenshots/create_role_manual_summary_page.png#center)

13. Click the `JSON` tab.

    ![Create Role Manual Policy Editor](../create_role_screenshots/create_iam_role_manual_policy_editor.png#center)

14. Bodo Cloud Platform requires a specific set of AWS permissions which
    are documented in [Bodo-Platform Policy](../downloadables/bodo-platform.json){target="blank"}.
    Paste the contents of the linked JSON file into the policy editor.

15. Click on `Review policy`.

16. In the *Name* field, add a policy name, e.g.
    `Bodo-Platform-User-Policy`. Click on `Create policy`.
    You will be taken back to the Role Summary.

17. From the role summary, copy the `Role ARN`. This is the value that
    you will enter into the **Role ARN** field on the *Setting* Page on
    the Bodo Platform.

    > ![Create Role Final Summary](../create_role_screenshots/create_iam_role_manual_final_summary.png#center)

#### Let the Bodo Platform create the IAM Role {#create_iam_role_using_platform}
***Let the Bodo Platform create the IAM Role***

1.  Follow the instructions from [AWS Account and Access Keys
    guide](https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html)
    to create/retrieve your AWS access key ID and secret access key.

2.  Click on `Create Role For Me` below the **Role ARN** field on the
    Setting page. This will open up a panel.

    ![Create Role Button on Platform](../create_role_screenshots/create_role_via_platform_create_role_button.png#center)

3.  Enter the Access Keys created in step 1 in the form and click on
    `CREATE ROLE`.

    ![Enter Access Keys to create role on Platform](../create_role_screenshots/create_role_via_platform_enter_access_keys.png#center)

    !!! note
         We will **not** save the provided Access Keys for security
        reasons.

4.  Click `OK` on the popup confirmation box.

5.  We will use the provided Access Keys to create an IAM Role in your
    AWS Account.

6.  The created Role ARN will be displayed on the same form.

    ![Role ARN generated on the Platform](../create_role_screenshots/create_role_via_platform_generated_role.png#center)

7.  Copy the generated Role ARN. This is the value that you will enter
    into the **Role ARN** field on the *Setting* Page on the Bodo
    Platform.

8.  In some cases, this role creation might fail. This could happen due
    to various reasons.

    a.  *A role already exists*: In this case, please open the [AWS
        Management Console](https://aws.amazon.com/console/), and
        navigate to the *IAM* Service. Click on *Roles* in the sidebar.
        Look for a Role named `BodoPlatformUser`. Click on
        the role, and copy over the **Role ARN** from the role summary.
        Alternatively, you can delete the existing role from the AWS
        Console and then try to create an IAM role again via the Bodo
        Platform. This will ensure you have the role set up with the
        correct permissions.

    !!! note 
        If this is a shared AWS Account, ensure that no one
        else is actively using this IAM Role before deleting it.

    b.  *Provided access keys are not valid*: Please ensure that valid
        access keys are provided.

    c. *Provided access keys don't have the right permissions to create
        a role*: Please ensure that the provided access keys have the
        permissions required to create an IAM Role.

    If none of these work, try creating the IAM Role manually as
    described in [earlier][create_iam_role_manually].

Once you have generated an IAM Role using either of the methods
described above, you are now ready to fill the Settings Form on the Bodo
Platform.

1.  Enter the Role ARN created using one of the above options into the
    **Role ARN** field in the Settings Form.
2.  Select a **Region** from the dropdown list. This is the region that
    your resources will be deployed in by default.
3.  Click on `SAVE`.

You can see the progress on granting
[AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html)
launch permissions to your account ID in the **AMI Share Status** field.
Your account is ready when it turns green.

!!! note
     We grant AMI launch permissions to your account in the
    following AWS regions: us-east-1, us-east-2, us-west-1 & us-west-2.

!!! important 
    We highly recommend that you ensure sufficient limits on
    your AWS account to launch resources. See
    [here][resources_created_in_aws_env] for details on the
    resources required for Bodo Cloud Platform.

!!! seealso "See Also"
    [Bodo Cloud Platform][bodo_platform]


## Resources Created in Your AWS Environment {#resources_created_in_aws_env}

Bodo deploys cluster/notebook resources in your own AWS environment to
ensure security of your data. Below is a list of AWS resources that the
Bodo Platform creates in your account to enable clusters and notebooks.

  AWS Service                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Purpose
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------
  [EC2 Instances](https://aws.amazon.com/ec2/)                                                                                                                                                                                                                                                                                                                                                                                                                                           |Cluster/notebook workers
  [EFS](https://aws.amazon.com/efs/)                                                                                                                                                                                                                                                                                                                                                                                                                                                     |Shared file system for clusters
  [VPC](https://aws.amazon.com/vpc/), [Subnets](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html), [NAT Gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html), [Elastic IP](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html), [ENI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-eni.html), [Security Groups](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html), ...   |Secure networking for clusters/notebooks
  [S3](https://aws.amazon.com/s3/) and [Dynamo DB](https://aws.amazon.com/dynamodb/)                                                                                                                                                                                                                                                                                                                                                                                                     |Resource states
  [AWS Systems Manager](https://aws.amazon.com/systems-manager/)                                                                                                                                                                                                                                                                                                                                                                                                                         |Managing EC2 instances
  [KMS](https://aws.amazon.com/kms/)                                                                                                                                                                                                                                                                                                                                                                                                                                                     |Cluster secrets (e.g. SSH keys)
  [IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) for Clusters                                                                                                                                                                                                                                                                                                                                                                                                |Allow cluster workers to access resources above

!!! note

    These resources incur additional AWS infrastructure charges and are not
    included in the Bodo Platform charges.


## AWS Account Cleanup {#aws_account_cleanup}

As explained [earlier][resources_created_in_aws_env], the platform creates two types of resources in the users'
AWS environments: organization level resources and cluster specific
resources. The organization level resources are created by the platform
to set up shared resources (such as a VPC, an EFS Mount, etc) that are
used later by all created resources. The cluster specific resources
(such as EC2 instances, ENIs, etc) are created by the platform to
host/manage a specific cluster. This includes notebooks and
corresponding resources as well. The cluster specific resources are
removed when you request a cluster to be removed. The organization level
resources persist in the user account so they can be used by clusters
deployed in the future. However, if you need to remove these resources
for any reason (AWS limits, etc.), an option to do so is provided.
Navigate to the *Settings* page and click on `Show Advanced` in the
bottom-right corner.

![Settings-Account-Cleanup](../platform_onboarding_screenshots/settings-ac-cleanup.png#center)

This will bring up a section called *AWS Resource Cleanup*.

![Advanced-Settings-Account-Cleanup](../platform_onboarding_screenshots/settings-adv-ac-cleanup.png#center)

Select the region from which you would like to remove these resources
(i.e. the region in which the resources you want to delete have been
created), and click `CLEANUP AWS RESOURCES`. Note that this will only
work if you don't have any active clusters in that region deployed
through the platform. Else, the request will be rejected, and you'll be
asked to remove all clusters in that region before trying again.
Removing active clusters (including clusters with a *FAILED* status) is
necessary because this process will make them inaccessible to the
platform.

!!! seealso "See Also"
    [Troubleshooting Managed Bodo Cloud Platform Issues on AWS][troubleshootingaws]


## Billing {#aws_billing}

Users subscribed to the Bodo Platform through the AWS Marketplace will
be charged for their use of the platform as part of their regular AWS
bill. The platform charges are based on the type of instances deployed
and the duration of their usage (to the nearest minute). The hourly rate
for the supported instance types can be found on our
[website](https://bodo.ai/pricing-aws). For any cluster deployed through
the platform, users are charged starting from when the cluster has been
successfully deployed, until the time the user requests the cluster to
be removed.

!!! note
    Users are not charged in case of failures in cluster creation.

As mentioned [previously][resources_created_in_aws_env], the AWS
resources set up by the platform in your AWS environment incur
additional AWS infrastructure charges, and are not included in the Bodo
Platform charges.
