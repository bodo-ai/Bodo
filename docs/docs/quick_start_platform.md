---
hide:
  - tags
  - navigation

tags:
  - getting started
  - platform

---

# Getting started with the Bodo Platform {#bodoplatformquickstart}


This page provides a quick start guide to the Bodo platform and explains its
important concepts briefly. We strongly recommend reading this page
before getting started with the Bodo hosted workspace.

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


##  Sign up for the Bodo Platform {#platformsignup}

1. Navigate to [Bodo Platform](https://platform.bodo.ai){target="blank"}. If
   you’re an AWS Customer, you can also sign up through the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-zg6n2qyj5h74o){target="blank"}.


2. Create a Bodo account using your preferred method; you can either use a social login
   via Github, Google or Microsoft or sign up with your Email.
   ![Sign Up Page](quick_start_screenshots/signup.png#center)


### Bodo Dashboard

1. After signing up, you’ll be directed to the dashboard, where you’ll see that a Community Edition Workspace has been assigned to your organization.
   ![Workspace view](quick_start_screenshots/workspaces.png#center)

2. Your workspace view contains all your workspaces. Find the *Community Edition Workspace* and click on the *Enter* button
   to enter the workspace.

!!! note  
    Usually, a Community Edition Workspace will be provisioned for your account immediately post sign up. On the rare occasion that you don’t see one on your dashboard, try refreshing the page.


  Once inside the Workspace, navigate to the *Notebooks* tab.
   ![Notebook view](quick_start_screenshots/notebook.png#center)


That’s it, you’re all set to experience Bodo. Follow along one of our tutorials or go through the curated list of bodo-examples.
 See `bodo-examples/hosted-trial-examples` for a set of notebooks ready to be run in your free trial environment.

## Community Edition Workspace Limitations

Community Edition Workspaces are one of the best ways to get started with Bodo.
However, Community Edition Workspaces have certain limitations.

- A community edition workspace is limited to a fixed quota of 30 compute hours per month.
- You can delete and provision new clusters and notebooks, but at any given time community edition workspaces are limited to only one notebook/cluster per workspace.
- All Trial Workspaces come with a preconfigured `2 x c5.2xlarge` cluster. You will not be able to configure this, say for example, to select a larger instance size for the cluster.
- Clusters on Trial Workspaces have no support for Jobs.

We recommend upgrading to Enterprise workspaces to bypass these limitations.  
You can [contact us](https://bodo.ai/contact){target="blank"} to get access to Enterprise workspaces,
or subscribe through [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-zg6n2qyj5h74o){target="blank"}.

!!! seealso "See Also"

    - [Bodo Platform Concepts][bodo_platform_concepts]
    - [Creating Clusters on the Bodo Platform][creating_clusters]
    - [Developers Quick Start Guide][devguide]
