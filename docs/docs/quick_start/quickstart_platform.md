---
hide:
  - tags

tags:
  - getting started
  - platform

---

# Getting started with the Bodo Cloud Platform {#bodoplatformquickstart}


This page provides a quick start guide to set up Bodo platform in your AWS account easily and quickly.


## AWS Marketplace Registration

a.  Subscribe through the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/B08NY29SMQ){target="blank"}.

!!! tip "Important"
    Make sure you have the necessary permissions to subscribe to the product in the AWS Marketplace. You need at least [`AWSMarketplaceManageSubscriptions`](https://docs.aws.amazon.com/marketplace/latest/buyerguide/buyer-iam-users-groups-policies.html#buyer-iam-builtin-policies){target="blank"} to manage subscriptions on AWS Marketplace. If you are not able to subscribe, please contact your AWS account administrator to get the necessary permissions.


b.  After confirming your subscription, click **Set up your Account** at the top right corner of the page. Bodo Platform's registration page will open in a new tab.

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
    Clicking on the confirmation link will take you to the Bodo Platform
    page where you can use your newly created credentials to sign in:
    ![Login-Page](../platform_onboarding_screenshots/login.png#center)


## AWS CloudFormation Quick Start

To be able to use the Bodo Platform to launch clusters and notebooks,
you must grant it permission to access your AWS account and provision
the required resources in it. These permissions are stored in the Bodo Platform using a "Cloud Configuration".
We provide an AWS CloudFormation template that you can execute in your AWS account,
which will create the necessary resources to support the Cloud Configuration in the Bodo Platform.

!!! tip Important
    You need the following set of permissions to successfully create the resources defined in the CloudFormation stack:

    - [`AWSCloudFormationFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AWSCloudFormationFullAccess.html){target="blank"}
    - [`AWSIAMFullAccess`](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/IAMFullAccess.html){target="blank"}

Once you have ensured that you have all permissions necessary to create the resources, follow the steps below to create a Cloud Configuration:

1. Access the *Cloud Configuration* page from the left sidebar. Then, click on *Create Cloud Configuration* in the top right corner of the page as shown in the picture below:

    ![Dashboard](../platform_onboarding_screenshots/dashboard.gif#center)

2. Select "AWS" as the cloud provider, and click on "Cloud Formation Template".

    !!! tip "Important"
        Do not close the form until you have completed all the steps following this one.

3. Fill in the following values :
    - **Cloud Configuration Name**: A name for your Cloud Configuration.
    - **CloudFormation Stack Region**: Fill this with the region where you want to deploy the stack.

4. Click on **Launch CloudFormation Template**. This will open the AWS CloudFormation console in a new tab in the selected region.


    !!! note "Important"
        All the values are pre-filled in the CloudFormation template. Please do not change any values.

5. Click on "Create Stack" to create the stack. This will create the necessary resources in your AWS account.
    ![CloudFormation Page](../platform_onboarding_screenshots/cloudformation.png#center)


    !!! note "Important"
        The stack creation process may take a few minutes to complete. Please wait until the stack is created successfully.

6. You can check the status of the stack from Bodo Platform as shown below. Once the stack is created successfully, Cloud Configuration will be created.

    ![CloudFormation Status](../platform_onboarding_screenshots/cloudformation-status.gif#center)



## Bodo Workspace Creation

Once your Cloud Configuration has been created, navigate to the *Workspaces* tab and click the *Create Workspace* button in the
 top right corner. Once your workspace has finished creating, you will be able to enter it.
 ![Dashboard view](../quick_start_screenshots/dashboard.png#center)

  Once inside the Workspace, navigate to the *Notebooks* tab.
   ![Notebook view](../quick_start_screenshots/notebook.png#center)


That’s it, you’re all set to experience Bodo. Follow along one of our tutorials or go through the curated list of bodo-examples.
 See `bodo-examples` for a set of notebooks ready to be run in your free trial environment.

!!! seealso "See Also"
    If you want to get started on using Bodo on Azure, please refer to the [Azure Platform Setup][bodo_platform_azure] guide.



## Basic Terminology

###  **Notebooks** {#jupyter-notebooks}

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
