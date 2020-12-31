.. _bodo_platform:

Bodo Cloud Platform
===================

This page descibes how to use the Bodo Cloud Platform, including registration, cluster creation, notebook attachment, and running jobs.


Registration
------------

a. Follow the registration link provided by a Bodo team member.
b. Fill out the fields with your information. If this is your individual account,
   use a unique name such as `firstname_lastname` for the Organization Name field.
c. Check the box for accepting terms and conditions and click on `SIGN UP`:

    .. image:: platform_onboarding_screenshots/signup.png
        :align: center
        :alt: Signup-Page

d. A page confirming that an activation link was sent to your email will appear.
   Please open the email and click on the activation link:

    .. image:: platform_onboarding_screenshots/signup-conf.png
        :align: center
        :alt: Signup-Page-Confirmation

  Clicking on the confirmation link will take you to the bodo platform page
  where you can use your newly created credentials to sign in:

        .. image:: platform_onboarding_screenshots/login.png
            :align: center
            :alt: Login-Page

.. _setting_aws_credentials:

Setting AWS Credentials
-----------------------

The next step is to link your AWS account to the Bodo platform. This can be done either using the *Settings* page
in the left bar or the first item in the *Onboarding* list highlighted in green as shown in the picture below:

.. image:: platform_onboarding_screenshots/dashboard.png
    :align: center
    :alt: Dashboard

Follow the instructions from `AWS Account and Access Keys guide <https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html>`_
to create/retrieve your AWS access key and access secret key, and `AWS Account ID guide <https://docs.aws.amazon.com/IAM/latest/UserGuide/console_account-alias.html>`_
to retrieve your AWS account ID.

Bodo Cloud Platform requires a minimal set of AWS permissions which are documented in :download:`Minimal Bodo-Platform Policy <downloadables/bodo-platform.json>`.

1. Create a policy from this :download:`Minimal Bodo-Platform Policy <downloadables/bodo-platform.json>` using the AWS-CLI as follows:

.. code-block:: bash

    aws iam create-policy --policy-name {POLICY_NAME} --policy-document file://{path-to-bodo-platform.json-file}

For instance:

.. code-block:: bash

    aws iam create-policy --policy-name bodo-platform --policy-document file:///Users/johndoe/Downloads/bodo-platform.json

2. Attach this policy to the specific user or group that will be used to generate the keys for the Bodo platform (using the AWS console for instance, in IAM --> Access Management --> Policies).


Enter your created credentials and account ID and click on *SAVE*.
Refresh and you can see the progress on granting `AMI <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>`_
launch permissions to your account ID. Your account is ready when it turns green.


**Note:** It is highly recommended that you ensure sufficient limits on your AWS account to launch
resources. See :ref:`resources_created_in_aws_env` for the resources required for Bodo Cloud Platform.


.. _creating_clusters:

Creating Clusters
-----------------

In the left bar click on *Clusters* (or click on the second step in the *Onboarding* list):

.. image:: platform_onboarding_screenshots/side-clusters.png
    :align: center
    :alt: Sidebar-Clusters
    :scale: 25

This will take you to the *Clusters* page. At the top right corner, click on
*Create Cluster* which opens the cluster creation form. First, choose a name for your cluster and
check the `EFA <https://aws.amazon.com/hpc/efa/>`_ checkbox if you want to use EFA-enabled nodes.
Then, select the type of nodes in the cluster to be created from the *Instance type* dropdown list.

**Note:** If the *Instance type* dropdown list does not populate, either the AWS
credentials are not entered properly or they are not valid.
Please go back to :ref:`setting_aws_credentials` and make sure you complete it with valid credentials.

Next, enter the number of nodes for your cluster in *Number of Instances*.
and choose the Bodo Version to be installed on your cluster. Typically the three latest Bodo Releases
are available.

**Note:** If the *Bodo Version* dropdown list does not populate, either the AWS
credentials are not entered properly or the permissions to Bodo's AMIs have not been granted to your account.
Please go back :ref:`setting_aws_credentials` and make sure you complete it with valid credentials and that
AMIs have been successfully shared with your AWS account.

Then, select a value for *Cluster auto shutdown*. This is the amount of time of inactivity after which
the platform will remove the cluster automatically. Activity is determined through attached notebooks (see :ref:`attaching_notebook_to_cluster`) 
and jobs (see :ref:`running_a_job`). Therefore, if you don't plan to attach a notebook or a job to this cluster 
(and use it via `ssh` instead), it's recommended to set this to `Never`, since otherwise the cluster will 
be removed after the set time.

.. image:: platform_onboarding_screenshots/cluster-form.png
    :align: center
    :alt: Cluster-creation-form

Finally click on `CREATE`.
You will see that a new task for creating the cluster has been created.

.. image:: platform_onboarding_screenshots/cluster-status-new.png
    :align: center
    :alt: Cluster-Status-New

The status is updated to *INPROGRESS* when the task starts executing and cluster creation is in progress.

.. image:: platform_onboarding_screenshots/cluster-status-ip.png
    :align: center
    :alt: Cluster-Status-InProgress

You can click on the *Details* drop down to monitor the progress for the cluster creation.

.. image:: platform_onboarding_screenshots/cluster-info.png
    :align: center
    :alt: Cluster-Info

Once the cluster is successfully created and ready to use, the status is updated to *FINISHED*.

.. image:: platform_onboarding_screenshots/cluster-status-done.png
    :align: center
    :alt: Cluster-Status-Finished

.. _attaching_notebook_to_cluster:

Attaching a Notebook to a Cluster
---------------------------------

Go to the notebooks page by clicking on *Notebooks* in the left bar (or on the third green step in the *Onboarding* list at the top).

.. image:: platform_onboarding_screenshots/side-nbs.png
    :align: center
    :alt: Sidebar-Notebooks
    :scale: 25

This will take you to the *Notebooks* page. At the top right corner, click on the *Create Notebook* button which opens
the notebook creation form.
Choose a name for your notebook and select
the type of node that will host the notebook
from the *Instance type* drop down list.
Note that this node is for running the Jupyter notebook itself, and will not run cluster workloads.
Lastly, select a cluster for attaching the notebook from the *Cluster* drop down menu and and click on `CREATE`.

.. image:: platform_onboarding_screenshots/nb-form.png
    :align: center
    :alt: Notebook-Creation-Form

After clicking `CREATE`, a new task for creating the notebook and its corresponding node is created.

.. image:: platform_onboarding_screenshots/nb-status-new.png
    :align: center
    :alt: Notebook-Status-New

The status updates to *INPROGRESS* when the task starts executing.

.. image:: platform_onboarding_screenshots/nb-status-ip.png
    :align: center
    :alt: Notebook-Status-InProgress

After creating the notebook, the platform runs AWS readiness probe checks:

.. image:: platform_onboarding_screenshots/nb-status-rp.png
    :align: center
    :alt: Notebook-Status-ReadinessProbe

The notebook is ready to use after all checks are complete.
*OPEN NOTEBOOK* will open the notebook in the current browser page,
while the dropdown allows opening the notebook in a new tab.

.. image:: platform_onboarding_screenshots/nb-status-done.png
    :align: center
    :alt: Notebook-Status-Finished

.. _running_a_job:

Running a Job
-------------

Bodo Cloud Platform has support for running scheduled (and immediate)
Python jobs without the need for Jupyter Notebooks. To create a Job, navigate
to the Jobs page by selecting `Jobs` in the left bar.

.. image:: platform_onboarding_screenshots/side-jobs.png
    :align: center
    :alt: Sidebar-Jobs
    :scale: 25

This pages displays any *INPROGRESS* jobs you have previously
scheduled and allows you to schedule new Jobs. At the top right corner, click on
`CREATE JOB`. This opens a job creation form. 

First, select a name for your job and specify the cluster on
which you want to deploy your job. If you have an existing cluster
that is not currently bound to a notebook or another job, you can select this cluster from the dropdown menu.
Alternatively, you can create a cluster specifically for this job by selecting
the `NEW` button next to the cluster dropdown menu. When creating 
a cluster specifically for a job, note that the cluster is only used for that job 
and is removed once the job completes. After selecting your cluster, indicate when you want your job 
to be executed in the `Schedule` section. Then, enter the `Command` that you want to execute inside this cluster.

**Note:** This command is automatically prepended with ``mpiexec -n <CORE_COUNT> python``. For example, 
to run a file ``ex.py`` with the argument 1, you would enter the command ``ex.py 1``.

To specify your source code location, fill in the `Path` line with a valid Git URL or S3 URI
that leads to a repository containing your code. 

**Note:** When selecting a GitHub URL, you should select the URL available at the top of your web browser
and NOT the path when cloning the repository, *i.e.* your path SHOULD NOT end in `.git`. If selecting an S3 URI,
your S3 bucket must be in the same region as your cluster.


.. image:: platform_onboarding_screenshots/jobs-form-standard.png
    :align: center
    :alt: Jobs-Forms-Standard


If you are cloning a private repository, you need to provide the platform with valid Git credentials to download your repository.
To do so, select `Show advanced` in the bottom right of the form. Then in `Workspace username`, enter your Git
username and in `Workspace password` enter either your password or a valid Github Access Token. The advanced options
also allow you to specify a particular commit or branch with `Workspace reference` and to load other custom environment
variables in `Other`.

**Note:** If your Github Account uses 2FA please use a Github Access Token to avoid any possible authentication issues.

Once your form is complete, select `CREATE` to begin your job. 

.. image:: platform_onboarding_screenshots/jobs-form-advanced.png
    :align: center
    :alt: Jobs-Forms-Advanced


Once you've provided all the necessary details, select `CREATE` to begin your job. You will see a *NEW* task
created in your jobs page.


.. image:: platform_onboarding_screenshots/jobs-new.png
    :align: center
    :alt: New-Job


If you created a cluster specifically for this job, a new cluster
will also appear in your clusters page.


.. image:: platform_onboarding_screenshots/jobs-cluster-inprogress.png
    :align: center
    :alt: New-Job-Cluster


Your job will begin once it reaches its scheduled time and any necessary clusters have been created.
Then your job will transition to being *INPROGRESS*.


.. image:: platform_onboarding_screenshots/jobs-inprogress.png
    :align: center
    :alt: InProgress-Job


At this point your job will execute your desired command. Once it finishes executing,
your job will transition to *FINISHED* status. You can find any stdout information 
that you may need by pressing `DETAILS` followed by `SHOW LOGS`. If a cluster was
specifically created for this job, it will be deleted after the job finishes.


.. image:: platform_onboarding_screenshots/jobs-finished.png
    :align: center
    :alt: Finished-Job


**Note:** Bodo DOES NOT preserve artifacts written to local storage. If you have any information that
you need to persist and later review, you should write to external storage, such as Amazon S3.
You may also write to stdout/stderr, but output logs may be truncated,
so it should not be considered reliable for large outputs that need to be read later.

.. _resources_created_in_aws_env:

Resources Created in Your AWS Environment
-----------------------------------------

Bodo deploys cluster/notebook resources in your own AWS environment to ensure
security of your data.
Below is a list of AWS resources
that the Bodo Platform creates in your account to enable clusters and notebooks.


.. list-table::
  :header-rows: 1

  * - AWS Service
    - Purpose
  * - `EC2 Instances <https://aws.amazon.com/ec2/>`_
    - Cluster/notebook workers
  * - `EFS <https://aws.amazon.com/efs/>`_
    - Shared file system for clusters
  * - `VPC <https://aws.amazon.com/vpc/>`_, `Subnets <https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html>`_,
      `NAT Gateway <https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html>`_,
      `Elastic IP <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html>`_,
      `ENI <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-eni.html>`_,
      `Security Groups <https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html>`_, ...
    - Secure networking for clusters/notebooks
  * - `S3 <https://aws.amazon.com/s3/>`_ and `Dynamo DB <https://aws.amazon.com/dynamodb/>`_
    - Resource states
  * - `AWS Systems Manager <https://aws.amazon.com/systems-manager/>`_
    - Managing EC2 instances
  * - `KMS <https://aws.amazon.com/kms/>`_
    - Cluster secrets (e.g. SSH keys)
  * - `IAM Role <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`_ for Clusters
    - Allow cluster workers to access resources above

.. note::

    These resources incur additional AWS infrastructure charges and are not included in the Bodo Platform charges.

.. _aws_account_cleanup:

AWS Account Cleanup
-------------------

As explained in :ref:`resources_created_in_aws_env`, the platform creates two types of resources in the users' AWS environments: 
organization level resources and cluster specific resources. The organization level resources are created by the platform to set 
up shared resources (such as a VPC, an EFS Mount, etc) that are used later by all created resources. The cluster specific resources 
(such as EC2 instances, ENIs, etc) are created by the platform to host/manage a specific cluster. This includes notebooks and 
corresponding resources as well.
The cluster specific resources are removed when you request a cluster to be removed.
The organization level resources persist in the user account so they can be used by clusters deployed in the future.
However, if you need to remove these resources for any reason (AWS limits, etc.), an option to do so is provided.
Navigate to the *Settings* page and click on `Show Advanced` in the bottom-right corner. 

.. image:: platform_onboarding_screenshots/settings-ac-cleanup.png
    :align: center
    :alt: Settings-Account-Cleanup


This will bring up a section called `AWS Resource Cleanup`. 

.. image:: platform_onboarding_screenshots/settings-adv-ac-cleanup.png
    :align: center
    :alt: Advanced-Settings-Account-Cleanup


Select the region from which you would like to remove these resources
(i.e. the region in which the resources you want to delete have been created), and click `CLEANUP AWS RESOURCES`.
Note that this will only work if you don't have any active clusters in that region deployed through the platform.
Else, the request will be rejected, and you'll be asked to remove all clusters in that region before trying again.
Removing active clusters (including clusters with a *FAILED* status) is necessary because 
this process will make them inaccessible to the platform.

.. _troubleshooting:

Troubleshooting
---------------

Here are solutions to potential issues you may encounter while using the Bodo Cloud Platform:

Cluster Creation Fails
~~~~~~~~~~~~~~~~~~~~~~

Most of cluster creation failures are usually due to one of the following:

- Your account hits AWS resource limits such as limits on the number of VPCs and EC2 instances
- Your AWS credentials do not have the required permissions (see :ref:`setting_aws_credentials`)
- AWS does not have enough of the requested resources (such as some of the large EC2 instances)

In case of failure, the logs are made available on the platform and should provide some details regarding why the failure occurred. Even though cluster creation was not successful, some AWS resources may still
have been provisioned. Click on the delete icon to remove all the created resources, otherwise you may incur charges for the provisioned AWS resources. 
You can try to create a cluster again after addressing the underlying issue such as increasing limits or providing AWS credentials with the required permissions.

Cluster Deletion Fails
~~~~~~~~~~~~~~~~~~~~~~

Failures during cluster deletion are very rare and usually only occur when the provisioned resources have been manually modified in some way.
In these cases, logs are provided to help you 
diagnose the issue. For instance, if logs indicate that some resource cannot be deleted due to a dependent resource, you can try to delete 
the resource manually through the 
`AWS Management Console <https://aws.amazon.com/console/>`_ and try to remove the cluster through the platform again.

Cleanup Shared Resources Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described in :ref:`aws_account_cleanup`, an option to remove organization level shared resources provisioned by Bodo in your AWS environment
is provided. If you need to remove resources manually (e.g. the process fails),
below is the list of organization level resources and the order to remove them.

**Note:** Please ensure that you have removed all clusters and related resources before proceeding. Deleting the resources
listed below may result in the platform losing access to those clusters for removal in the future.

The resources should be easy to identify within their respective
sections on the `AWS Management Console <https://aws.amazon.com/console/>`_ since their names are all prefixed with `bodo`.

1. Navigate to the `AWS Management Console <https://aws.amazon.com/console/>`_. Sign in if you are not already signed in. Make sure you have selected
   the region from which you want to remove the shared resources.

2. Click on `Services` in the top-right corner. Navigate to the `EC2` section (under `Compute`) and then to `Network Interfaces` in the sidebar 
   (under `Network & Security`). You will see two Network Interfaces. One of them is required for an EFS Mount (shared storage),
   and the other is required by a NAT Gateway. These dependent resources need to be removed first.
    
   a.  Click on `Services` and navigate to the `EFS` section (under `Storage`). Click on `File Systems` in the sidebar. Delete the File System
       prefixed with `bodo` by selecting it and clicking on `Delete`.

   b.  Click on `Services` and navigate to the `VPC` section (under `Networking & Content Delivery`). Select `NAT Gateways` in the 
       sidebar (under `Virtual Private Cloud`). Select the NAT Gateway prefixed with `bodo` and delete it.
   
   Navigate back to `Network Interfaces` in the `EC2` section and ensure that the two ENIs are deleted (or have the status `available`). 
   This may take a few minutes in some cases.

3. Click on `Services` and navigate to the `VPC` section (under `Networking & Content Delivery`). Select `Your VPCs` in the 
   sidebar (under `Virtual Private Cloud`). Select the VPC prefixed with `bodo` and delete it. If there is a dependency warning,
   wait for a few minutes and try again. You can also try to delete the linked dependent resources manually if it does not resolve on its own.

4. Finally, click on `Services` in the top-right corner. Navigate to the `EC2` section (under `Compute`) and select `Elastic IPs` in the sidebar
   (under `Network & Security`). Select the EIP prefixed with `bodo` and select `Release Elastic IP addresses` under `Actions`.

The steps above should remove the organization level resources provisioned by Bodo in your AWS environment.

.. _aws_billing:

Billing
-------

Users subscribed to the Bodo Platform through the AWS Marketplace will be charged for their use of the platform as part
of their regular AWS bill. The platform charges are based on the type of instances deployed and the duration of their usage (to the nearest minute).
The hourly rate for the supported instance types can be found on our `website <https://bodo.ai/aws-pricing>`_.
For any cluster deployed through the platform, users are charged starting from when the cluster has been successfully deployed, 
until the time the user requests the cluster to be removed. 

**Note:** Users are not charged in case of failures in cluster creation.

As mentioned previously in :ref:`resources_created_in_aws_env`, the AWS resources set up by the platform in your AWS environment
incur additional AWS infrastructure charges, and are not included in the Bodo Platform charges.
