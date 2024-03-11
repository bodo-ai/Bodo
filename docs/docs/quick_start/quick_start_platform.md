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


##  Setup the Bodo Platform {#platformsetup}

Follow the [Bodo Platform Setup Guide][bodo_platform_aws].

If you sign up for Bodo through the AWS marketplace, you get a 14-day free trial. For the duration of your trial, you will only be charged for the underlying AWS resources created by your
    activity on the Bodo Platform. After the trial expires you will be charged according to our [pay-as-you-go pricing](https://www.bodo.ai/pricing)

## AWS CloudFormation Quick Start

If you are using AWS, you can follow the instructions below to get quickly onboarded to the Bodo Platform. First, you need to subscribe to Bodo through the [AWS marketplace](https://aws.amazon.com/marketplace/pp/prodview-zg6n2qyj5h74o) and complete registration on the Bodo Platform. 

Then, navigate to the cloud configuration page.  
![Dashboard](../platform_onboarding_screenshots/dashboard.gif#center)

To be able to use the Bodo Platform to launch clusters and notebooks,
you must grant it permission to access your AWS account and provision
the required resources in it.

1. Click on the "Create Cloud Configuration" button to create a new cloud configuration.
2. Then select "AWS" as the cloud provider, and click on "Manual" to enter the required information manually.

    !!! tip "Important"
        Do not close the dialog box until you have completed all the steps following this one.

3. Fill in the following values :
    - **Cloud Configuration Name**: A name for your cloud configuration.
    - **S3 Bucket**: Fill this with the value `bodo-default-s3-bucket-<external_ID>`, where `<external_ID>` is the external ID generated in the form right above this field.
    - **Dynamo DB Table**: Fill this with the value `bodo-default-dynamodb`. 
    - **Role ARN**: Fill this with the value `arn:aws:iam::<aws_account_id>:role/BodoPlatformUser`, where `<aws_account_id>` is your AWS account ID.
    - **Metadata Storage Region**: Fill this with the region where you want to store the metadata.
4. Now navigate to this link: [Bodo Platform AWS Setup](https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?templateURL=https://platform-quickstart.s3.amazonaws.com/bodo_platform_default_quickstart.yml&stackName=bodo-default-stack&param_MetadataRegion=enter_your_metadata_region_here&param_BodoExternalId=enter_your_external_id_here). Fill in the required parameter values as indicated.

    !!! tip "Important" 
        - The `MetadataRegion` parameter should be the same as the `Metadata Storage Region` you entered in the previous step. 
        - The `BodoExternalId` parameter should be the same as the `External ID` generated in the form in the previous step.

### Bodo Dashboard

1. Once your cloud config has been created navigate to the *Workspaces* tab and click the *Create Workspace* button in the
 top right corner. Once your workspace has finished creating, you will be able to enter it.
 ![Dashboard view](../quick_start_screenshots/dashboard.png#center)

  Once inside the Workspace, navigate to the *Notebooks* tab.
   ![Notebook view](../quick_start_screenshots/notebook.png#center)


That’s it, you’re all set to experience Bodo. Follow along one of our tutorials or go through the curated list of bodo-examples.
 See `bodo-examples` for a set of notebooks ready to be run in your free trial environment.
