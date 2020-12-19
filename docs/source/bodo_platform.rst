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
Please go back :ref:`setting_aws_credentials` and make sure you complete it with valid credentials.

.. image:: platform_onboarding_screenshots/cluster-form.png
    :align: center
    :alt: Cluster-creation-form

Finally, enter the number of nodes for your
cluster in *Number of Instances* and click on `CREATE`.
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

Attaching a Notebook to a Cluster
---------------------------------

Go to the notebooks page by clicking on *Notebooks* in the left bar (or on the third green step in the *Onboarding* list at the top).

.. image:: platform_onboarding_screenshots/side-nbs.png
    :align: center
    :alt: Sidebar-Notebooks
    :scale: 25

This will take you to the *Notebooks* page. At the top right corner, click on the *Create Notebook* which opens
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
