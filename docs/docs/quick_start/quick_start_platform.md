---
hide:
  - tags

tags:
  - getting started
  - platform

---

# Getting started with the Bodo Platform {#bodoplatformquickstart}


This page provides a quick start guide to the Bodo platform and explains its
important concepts briefly. We strongly recommend reading this page
before getting started with the Bodo platform.

## Basic Terminology

###  **Notebooks**

  A Notebook is a simple instances of a JupyterLab Server. You can use
  a notebook to navigate the files in your workspace, define and execute
  your workflows on the clusters, etc.


### **Clusters**

  A Cluster provides your notebooks with necessary compute resources.
  Each cluster is tied to a specific Bodo version. A cluster also has
  additional configuration parameters such as the number of instances,
  auto-pause duration, etc. You can read more about creating a cluster [here][creating_clusters].

### **Workspaces**

  Workspaces are a basic unit of organization on Bodo platform. You can
  think of a workspace as a directory on a UNIX file system. Each Workspace
  contains zero or more Notebooks, Clusters, etc.


## Registration

!!! tip "Important"
    Make sure you have the necessary permissions to subscribe to the product in the AWS Marketplace. You need at least [`AWSMarketplaceManageSubscriptions`](https://docs.aws.amazon.com/marketplace/latest/buyerguide/buyer-iam-users-groups-policies.html#buyer-iam-builtin-policies){target="blank"} to manage subscriptions on AWS marketplace. If you are not able to subscribe, please contact your AWS account administrator to get the necessary permissions.

a.  Subscribe through the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/B08NY29SMQ){target="blank"}.

b.  After confirming your subscription, you should click **Set up your Account** at the top right corner of the page. Bodo Platform's registration page will open in a new tab.

![Set up Account](../platform_onboarding_screenshots/set-up-account.png#center)


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


## AWS CloudFormation Quick Start

To be able to use the Bodo Platform to launch clusters and notebooks,
you must grant it permission to access your AWS account and provision
the required resources in it. These permissions are stored in the Bodo Platform using a "Cloud Configuration". We provide an AWS CloudFormation template that you can execute in your AWS account, which will create the necessary resources to support the cloud configuration in the Bodo Platform.

To create a cloud configuration on the Bodo Platform, access the *Cloud Configuration* page from the left sidebar. Then, click on *Create Cloud Configuration* at the top right corner of the page as shown in the picture below:

![Dashboard](../platform_onboarding_screenshots/dashboard.gif#center)


!!! tip Important
    You need the following set of permissions to successfully create the resources defined in the CloudFormation stack:

    - [`AWSCloudFormationFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AWSCloudFormationFullAccess.html){target="blank"}
    - [`AmazonS3FullAccess`](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-iam-awsmanpol.html#security-iam-awsmanpol-amazons3fullaccess){target="blank"}
    - [`AmazonDynamoDBFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonDynamoDBFullAccess.html){target="blank"}
    - [`AWSIAMFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/IAMFullAccess.html){target="blank"}

Once you have ensured that you have all permissions necessary to create the resources, follow the steps below to create a cloud configuration:

1. Click on the "Create Cloud Configuration" button to create a new cloud configuration.
2. Then select "AWS" as the cloud provider, and click on "Manual" to enter the required information manually.

    !!! tip "Important"
        Do not close the form until you have completed all the steps following this one.

2. Note the external id generated in the form. You will need this value in the next step.
    
    ![Dashboard view](../platform_onboarding_screenshots/external_id.png#center)

3. Fill in the following values :
    - **Cloud Configuration Name**: A name for your cloud configuration.
    - **S3 Bucket**: Fill this with the value `bodo-default-s3-bucket-<external_ID>`, where `<external_ID>` is the external ID generated in the form right above this field.
    - **Dynamo DB Table**: Fill this with the value `bodo-default-dynamodb-<external_ID>`. 
    - **Role ARN**: Fill this with the value `arn:aws:iam::<aws_account_id>:role/BodoPlatformUser`, where `<aws_account_id>` is your AWS account ID.
    - **Metadata Storage Region**: Fill this with the region where you want to store the metadata.
   
4. Now navigate to this link: [Bodo Platform AWS Setup](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?templateURL=https://platform-quickstart.s3.amazonaws.com/bodo_platform_default_quickstart.yml&stackName=bodo-default-stack&param_MetadataRegion=enter_your_metadata_region_here&param_BodoExternalId=enter_your_external_id_here){target=blank}. Fill in the required parameter values as indicated in the screenshot below.

    ![CloudFormation](../platform_onboarding_screenshots/cloudformation.png#center)

    !!! tip "Important" 
        - The `MetadataRegion` parameter should be the same as the `Metadata Storage Region` you entered in the previous step. 
        - The `BodoExternalId` parameter should be the same as the `External ID` generated in the form in the previous step.

5. After the stack is finished creating, navigate back to your cloud configuration creation form on the Bodo Platform and click on "Validate" the cloud configuration. If the validation is successful, you can click on "Create" to create the cloud configuration.


## Bodo Dashboard

1. Once your cloud config has been created navigate to the *Workspaces* tab and click the *Create Workspace* button in the
 top right corner. Once your workspace has finished creating, you will be able to enter it.
 ![Dashboard view](../quick_start_screenshots/dashboard.png#center)

  Once inside the Workspace, navigate to the *Notebooks* tab.
   ![Notebook view](../quick_start_screenshots/notebook.png#center)


That’s it, you’re all set to experience Bodo. Follow along one of our tutorials or go through the curated list of bodo-examples.
 See `bodo-examples` for a set of notebooks ready to be run in your free trial environment.

!!! seealso "See Also"
    If you want to get started on using Bodo on Azure, please refer to the [Azure Platform Setup][bodo_platform_azure] guide.

