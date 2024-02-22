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

1. Follow the [Bodo Platform Setup Guide][bodo_platform_aws].

If you sign up for Bodo through the AWS marketplace, you get a 14-day free trial. For the duration of your trial, you will only be charged for the underlying AWS resources created by your
    activity on the Bodo Platform. After the trial expires you will be charged according to our [pay-as-you-go pricing](https://www.bodo.ai/pricing)


### Bodo Dashboard

1. Once your cloud config has been created navigate to the *Workspaces* tab and click the *Create Workspace* button in the
 top right corner. Once your workspace has finished creating, you will be able to enter it.
 ![Dashboard view](../quick_start_screenshots/dashboard.png#center)

  Once inside the Workspace, navigate to the *Notebooks* tab.
   ![Notebook view](../quick_start_screenshots/notebook.png#center)


That’s it, you’re all set to experience Bodo. Follow along one of our tutorials or go through the curated list of bodo-examples.
 See `bodo-examples` for a set of notebooks ready to be run in your free trial environment.
