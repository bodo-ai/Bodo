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
platform. This can be done using the *Cloud Configuration* page in the left
bar as shown in the picture below:

![Dashboard](../platform_onboarding_screenshots/dashboard.gif#center)

To be able to use the Bodo Platform to launch clusters and notebooks,
you must grant it permission to access your AWS account and provision
the required resources in it. This can be done through two ways:

### 1. With Access Keys {#with Access Keys}

1.  Follow the instructions from [AWS Account and Access Keys
    guide](https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html)
    to create/retrieve your AWS access key ID and secret access key.

2.  Enter the Access Key ID and Secret created in the previous step.

3. Select the region the where the metadata will be stored and click on
    `CREATE`.

    ![Enter Access Keys on Platform](../create_role_screenshots/create_config_with_accesskeys.png#center)


    !!! note
         This has been tested using access keys that have `S3FullAccess`, `DynamoDBFullAccess` and `IAMFullAccess` permissions. 
         **Access keys with limited permissions will not work**.

    !!! note
         We will **not** save the provided Access Keys for security
        reasons.





### 2. Manual Process {#create_manually}

Open the *Cloud Configuration Form* and note down the `External ID`.

We need to create 3 resources through the AWS Console and provide details about them in 
the *Cloud Configuration Form*.

Before that, Open the Cloud Configuration Form and note down the External ID.

![Cloud-Configuration](../create_s3_and_dynamo_screenshots/cloud_config_id.png#center)

!!! note
   The S3 bucket and the Dynamo DB table should be created in the same region. 
   Supported regions are: 'us-east-1', 'us-east-2', 'us-west-1' and 'us-west-2'.

#### Setup S3 Bucket {#setup_s3_bucket}
- **Setup S3 Bucket**
    1. In AWS console, go to S3 Management Console and click on `Create Bucket`.
     ![S3-Page](../create_s3_and_dynamo_screenshots/s3_bucket_console.png#center)

    2. In the creation form, enter the **Bucket Name** and select the **AWS Region**. 
       Note these down since you will need to enter them in the *Cloud Configuration Form*.
    ![create-s3-Page](../create_s3_and_dynamo_screenshots/s3-create-page.png#center)

    3. All other fields can remain as it is by default and click on `Create Bucket`.

#### Setup Dynamo DB table {#setup_dynamo_db}
- **Setup Dynamo DB table**
    1. In AWS console, go to Dynamo DB page and click on `Create Table`.
     ![dynamo-Page](../create_s3_and_dynamo_screenshots/dynamo_console.png#center)
    
    2.  In the table creation form, enter the **Table name**. 
        Note this down since you will need to enter it in the *Cloud Configuration Form*.

    3. Under Partition key, enter the partion key as **LockID** and select the type as **String**. This partion key must be created
       for Bodo Platform to work.
     ![dynamo-create-page](../create_s3_and_dynamo_screenshots/create-dynamo.png#center)
     
    4. All other fields can remain as it is by default and click on `Create Table`.

#### Setup IAM role {iam_role_setup}
- **IAM role**

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
        *Cloud Configuration* form on the Bodo Platform.
        
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
    
    15. **`$$BUCKET_NAME$$`** and **`$$DYNAMO_ARN$$`** placeholders needs to be updated in the Bodo Platform Policy 
        with the **S3 Bucket name** and **Dynamo DB ARN** respectively.

    15. Click on `Review policy`.

    16. In the *Name* field, add a policy name, e.g.
        `Bodo-Platform-User-Policy`. Click on `Create policy`.
        You will be taken back to the Role Summary.

    17. From the role summary, copy the `Role ARN`. This is the value that
        you will enter into the **Role ARN** field on the *Setting* Page on
        the Bodo Platform.

        > ![Create Role Final Summary](../create_role_screenshots/create_iam_role_manual_final_summary.png#center)


Once you have generated an IAM Role using the steps described above, 
you can fill the remaining fields in the **Cloud Configuration** form on the Bodo
Platform.

1.  Enter the **Name** of the configuration.
2.  Enter the **S3 Bucket** name.
3.  Enter the **Dynamo DB Table** name. 
4.  Enter the Role ARN in the **Role ARN** field.
5.  Select a **Region** from the dropdown list. 
    This is the region of your S3 bucket and Dynamo DB table.
6.  Click on `CREATE`.


![Create Cloud Configuration AWS Manual ](../platform_onboarding_screenshots/cloud_configuration_aws_manual.png#center)


!!! important 
    We highly recommend that you ensure sufficient limits on
    your AWS account to launch resources. See
    [here][resources_created_in_aws_env] for details on the
    resources required for Bodo Cloud Platform.



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


!!! seealso "See Also"
    [Troubleshooting Managed Bodo Cloud Platform Issues on AWS][troubleshootingaws]



## Using Bodo Platform
Check the following link on how to use the Bodo Platform once the **cloud credentials are added**.

[Bodo Cloud Platform][bodo_platform]


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
### Billing Alarms
You can set up AWS alarms to monitor usage using cloudwatch alarms on AWS.

![Navigate-To-AWS-CloudWatch](../platform_onboarding_gifs/aws-alarms.gif#center)

#### Steps to create an alarm on your AWS account for all EC2 usage:
***Steps to create an alarm on your AWS account for all EC2 usage:***

![Create-AWS-Alarm](../platform_onboarding_gifs/create-alarm.gif#center)

1. Select the region from which you would like to create the alarm and click `Create Alarm`. 

2. Click on the `Select metric` which would bring you to a search bar that allows you 
to search and select the metric of your choice. Make sure to click on the check box of the 
metric of your choice. In this example, we choose vCPU for monitoring EC2 usage. 

3. Set a reasonable number for the threshold for an alarm to go off based on your usage expectations.
If you do not have this you can use the history of the metric to get an estimate.
The history can be viewed by clicking on the expand button on the graph.
You can toggle the time range by clicking on the available options (e.g. `1w` for 1 week)
on the top panel. You can use the graph to set the desired threshold for your alarm. The 
default period for the alarm threshold is 5 minutes but can be altered based on your requirement. 
In this example above we set the alarm to become active, if the vCPU count is greater than 1000 
for 5 minutes as the highest value found from the last week was ~850. Click `Next` at the bottom 
of the page after you have set the threshold and period. 

4. You will now be asked to select the Simple Notification Service (SNS) Topic for this alarm. 

    a. If you already have an existing SNS Topic you can choose it from the dropdown list 
    by clicking on the `Select an Existing SNS Topic` radio button. 

    b. If you do not have an SNS Topic, then create a new SNS Topic by clicking on the 
    `Create New Topic` radio button. Fill in the form with an appropriate `Topic Name` and 
    provide the emails (Those who should be alerted by the alarm) in the 
    `Email endpoints that will receive the notification` tab.

    Once you have provided these details you can click on `Next` at the bottom of the page.

5. You will now be required to fill out the details of the alarm itself, fill the fields with the 
appropriate details and click `Next` at the bottom of the page. 

6. Finally, preview your alarm before you click on `Create Alarm` at the bottom of the page to create the Alarm.
