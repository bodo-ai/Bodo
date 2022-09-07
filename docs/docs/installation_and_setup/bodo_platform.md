# Using Bodo Cloud Platform {#bodo_platform}

## Bodo Cloud Platform Concepts {#bodo_platform_concepts}

This page describes the fundamental concepts you need to know to use the Bodo Cloud Platform.

### Organizations

Organizations on the Bodo Cloud Platform are tenants for billing and cloud resource management purposes.
An organization can have multiple [workspaces][workspaces] and [cloud configurations][cloud-configurations], and users can be part of multiple organizations.

![User-Mgmt](../platform2-screenshots/user_mgmt.png#center)

### Cloud-Configurations

A *cloud-configuration* is an entity used to store information about your AWS or Azure account.
It consists of:

1. Details regarding the trust relationship between the platform and your cloud provider account.
   For AWS accounts, this is done through a cross-account IAM role.
   For Azure account, this is done through a service principal (scoped to a specific resource group)
   for the Bodo Platform application.
   This gives the platform the ability to provision and manage cloud resources in your account.
2. Details regarding metadata storage. The platform needs to store
   certain metadata to carry out its functions, such as the state of your various cloud deployments, logs, etc.
   On AWS, this data is stored in an S3 bucket and a DynamoDB table.
   On Azure, this data is stored in a storage container.


![Cloud-Configs](../platform2-screenshots/cloud_configs.png#center)

### Workspaces

A *workspace* on the Bodo Cloud Platform consists of:

1. A shared filesystem where you can collaborate with your team on your projects.
2. Networking infrastructure such as virtual networks, security groups and subnets in which
   your compute clusters and Jupyter servers will be securely deployed.

A workspace is tied to a particular cloud-configuration and has its own user-management i.e.,
you can have different subsets of users with different sets of roles
and permissions in different workspaces within the same organization.

!!! important
     If a user that is not part of the organization, is invited to a workspace in the organization,
    it is automatically added to the organization with minimal permissions.

![Workspaces](../platform2-screenshots/workspace.png#center)

To create a workspace, go to the "Workspaces" section in the sidebar and click on "Create Workspace". In the
creation form, enter the name of the workspace, select the cloud-configuration to use for provisioning
it and the region where it should be deployed, and click on "Create Workspace".

![Create-Workspace](../platform2-screenshots/create-workspace-form.png#center)

This will start the workspace deployment. When the workspace is in the "READY" state, click on
the button next to it to enter it.

![Enter-Workspace](../platform2-screenshots/enter-workspace.png#center)

## Notebooks

Jupyter servers act as your interface to both your shared file-system and your compute clusters.
Users can execute code from their notebooks on the compute cluster from the Jupyter interface.
A Jupyter server is automatically provisioned for your use when you first enter the workspace.

![Notebook-View](../platform2-screenshots/notebook_view.png#center)

You can view and manage all the Jupyter servers in the "Notebook Manager" section of
"Workspace Settings".

![Notebook-Manager](../platform2-screenshots/notebook_manager.png#center)

## Creating Clusters {#creating_clusters}

In the left bar click on _Clusters_ (or click on the second step in the
_Onboarding_ list). This will take you to the _Clusters_ page. At the top right corner,
click on `Create Cluster` which opens the cluster creation form.

![Cluster-Create](../platform2-gifs/create_cluster.gif#center)

Cluster creation form:

![Cluster-Form](../platform2-gifs/create_cluster_form.gif#center)

First, choose a name for your cluster.

Then, select the type of nodes in the cluster to be created from the **Instance type** dropdown
list. [EFA](https://aws.amazon.com/hpc/efa/){target="blank"} will be used if the instance type supports it.

![Cluster-Form-Instance](../platform2-gifs/create_cluster_list.gif#center)

!!! note
    If the **Instance type** dropdown list does not populate,
    either the credentials are not entered properly or they are not valid.
    Please see how to set your [AWS][setting_aws_credentials]
    or [Azure][setting_azure_credentials] credentials and make sure your credentials are valid.

Next, enter the number of nodes for your cluster in **Number of
Instances**. and choose the Bodo Version to be installed on your
cluster. Typically the three latest Bodo Releases are available.

![Cluster-Form-Bodo](../platform2-screenshots/cluster_bodo_version.png#center)


Then, select a value for **Cluster auto pause**. This is the amount
of time of inactivity after which the platform will pause the cluster
automatically.

![Cluster-Form-Auto-Pause](../platform2-screenshots/cluster_auto_pause.png#center)


Additionally, you can select a value for **Cluster auto shutdown**. Activity is determined through attached notebooks (see
[how to attach a notebook to a cluster][attaching_notebook_to_cluster]) and jobs
(see [how to run a job][running_a_job]). Therefore, if you
don't plan to attach a notebook or a job to this cluster (and use it
via `ssh` instead), it's recommended to set this to
`Never`, since otherwise the cluster will be removed after
the set time.

![Cluster-Form-Advanced](../platform2-screenshots/cluster_auto_shutdown.png#center)

Finally click on `CREATE`. You will see that a new task for creating the
cluster has been created. The status is updated to <inpg>INPROGRESS</inpg> when the task starts executing and
cluster creation is in progress.

![Cluster-Status-InProgress](../platform2-screenshots/cluster_inprogress.png#center)

You can click on the `Details` drop down to monitor the progress for the
cluster creation.

![Cluster-Info](../platform2-screenshots/cluster_inprogress_deatails.png#center)

Once the cluster is successfully created and ready to use, the status is
updated to <fin>FINISHED</fin>.

![Cluster-Status-Finished](../platform2-gifs/create_cluster_details.gif#center)

## Attaching a Notebook to a Cluster {#attaching_notebook_to_cluster}

To attach a notebook to a cluster, select the cluster from the drop-down in the top-left.

![Attach-Cluster](../platform2-gifs/attach-cluster.gif#center)

To execute your code across the attached cluster, use IPyParallel magics `%%px` and `%autopx`.

![Run-Code-Notebook](../platform2-gifs/parallel-magic.gif#center)

Note that parallel execution is only allowed when the notebook is attached to a cluster.
If you execute a cell without a cluster attached, the following warning will be shown:

![Detached-Notebook-Warning](../platform2-gifs/not-attached-to-cluster-warning.gif#center)

## Using your own Instance Role for a Cluster {#instance_role_cluster}
### AWS
In cases where you want to access additional AWS resources from Bodo clusters e.g. S3 buckets,
you can create an IAM Role in your AWS account and then register it as an Instance Role on the Bodo Platform which will allow you to access those resources from Bodo clusters without using AWS keys.

![View-Instance-Roles](../platform2-gifs/instance-role-list.gif#center)

Note that, by default, Bodo creates an IAM role with necessary policies for each cluster. When you register your own role with the Bodo Platform, it will automatically attach the other required policies to this role.

Here we walk through the process of setting up an IAM Role in AWS and then registering it as an Instance Role on the Bodo Platform. For this example, we will be creating a role with access to an S3 bucket in your AWS account:

Step 1: Create an AWS IAM Role on the [AWS Management Console](https://aws.amazon.com/console/):
1. Go to the IAM service.

![AWS-IAM](../platform2-screenshots/aws_iam.png#center)

2. In the left sidebar click on Roles.

![AWS-IAM-ROLE](../platform2-screenshots/aws_iam_role.png#center)

3. Click on button `Create role`, then select:
   * Trusted entity type: **AWS service**
   * Common use cases: **EC2**

![AWS-IAM-Role-Form](../platform2-screenshots/aws_iam_role_form.png#center)

4. Click next, and then create new policy that will be attached to this role:
   * json policy:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::<private-s3-bucket-name>"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObjectAcl"
            ],
            "Resource": [
                "arn:aws:s3:::<private-s3-bucket-name>/*"
            ]
        }
    ]
}
```

5. Go back to Create role, refresh list of policies and add the policy that was created.
6. Click **Next**, then in the Role name field, type a role name and click **Create role**.
7. Copy the Role ARN from the role summary.

![AWS-IAM-Role-ARN](../platform2-screenshots/aws_iam_role_arn.png#center)


Step 2: Register your AWS IAM Role on the Bodo Platform as a new Instance Role:

1. Click on the **CREATE INSTANCE ROLE** button and in the creation form, fill the following fields:
   * Name: Name for the Instance Role 
   * Role ARN: AWS Role ARN from Step 1
   * Description: Short description for Instance Role

![Create-Instance-Role](../platform2-screenshots/instance-role-form.png#center)


2. Click on the **Create** button.


The Instance Role will now be registered on the Platform. It can have one of two status-es:
* **Active**: Instance Role is ready to use
* **Failed**: Something went wrong while registering the Instance Role and it cannot be used. Some possible problems could be:
   * The Platform wasn't able to find the specified Role.
   * The Platform was not able to attach additional Bodo polices that are required for normal cluster operations.

## Managing Packages on the cluster using IPyParallel magics - Conda and Pip

We recommend all packages to be installed using Conda as that is what we use in our environments.
Any conda command can be run in parallel on all the nodes of your cluster using `%pconda`.
To install a new package on all the nodes of your cluster you can use `%pconda install`.
All conda install arguments work as expected, e.g. `-c conda-forge` to set the channel.

```shell
%pconda install -c conda-forge <PACKAGE_NAME>
```

To learn more about the packages installed on the cluster nodes `%pconda list`.
```shell
%pconda list
```

To remove a conda package on all the nodes of your cluster, use `%pconda remove`.

```shell
%pconda remove <PACKAGE_NAME>
```

![Conda-Magic](../platform2-gifs/conda-magic.gif#center)

Any pip command can be run in parallel on all the nodes of your cluster using `%ppip`.

Example:
```shell
%ppip install <PACKAGE_NAME>
```

To learn about the installed packages, you can use `%ppip show`
to get the details of the package.

```shell
%ppip show <PACKAGE_NAME>
```

To remove the same package on all the nodes of your cluster, use `%ppip uninstall`.

```shell
%ppip uninstall <PACKAGE_NAME> -y
```

![Pip-Magic](../platform2-gifs/pip-magic.gif#center)

## Running shell commands on the cluster using IPyParallel magics

Shell commands can be run in parallel on the nodes of your cluster using `%psh <shell_command>`.

```shell
%psh echo "Hello World"
```

![Shell-Magic](../platform2-gifs/shell-magic.gif#center)
## Connecting to a Cluster {#connecting_to_a_cluster}

We recommend interacting with clusters primarily through Jupyter
notebooks and Jobs. However, it may be necessary to connect directly to
a cluster in some cases. In that case, you can connect through a notebook
terminal.

### Connecting with a Notebook Terminal

First, you need to [create a cluster][creating_clusters]
and [attach a notebook to the cluster][attaching_notebook_to_cluster].
This will create the ssh-key at ``~/cluster_ssh_keys/id_rsa-<CLUSTER-UUID>``.

Then, go the cluster tab and find your cluster. Click on `DETAILS` and
copy the cluster `UUID` and `IP address` of the node you would like to connect to.

![Cluster-UUID-Info](../platform_onboarding_screenshots/cluster-ip-info.png#center)

Next, go to the notebooks tab and select `OPEN NOTEBOOK`. In the
_Launcher_, click on `Terminal`.

![Notebook-Terminal](../platform_onboarding_screenshots/notebook-terminal.png#center)

In the terminal you can connect to any of the cluster nodes by running
```shell
ssh -i ~/cluster_ssh_keys/id_rsa-<CLUSTER_UUID> <IP>
```

![Connect-Cluster](../platform2-gifs/connect-to-cluster.gif#center)

Through this terminal, you can interact with the `/shared` folder, which
is shared by all the instances in the cluster and the Notebook instance.
[Verify your connection][verify_your_connection] to interact directly with your cluster.

### Verify your Connection {#verify_your_connection}

Once you have connected to a node in your cluster, you should verify
that you can run operations across all the instances in the cluster.

1.  Verify the path to the hostfile for your cluster. You can find it by
    running:
    ```shell
    ls -la /shared/.hostfile-<CLUSTER UUID>
    ```
2.  Check that you can run a command across you cluster. To do this,
    run:

    ```shell
    mpiexec -n <TOTAL_CORE_COUNT> -f /shared/.hostfile-<CLUSTER UUID> hostname
    ```

    This will print one line per each core in the cluster, with one
    unique hostname per cluster node.

    Your cluster's `TOTAL_CORE_COUNT` is usually half the
    number of vCPUs on each instance times the number of instances in
    your cluster. For example, if you have a 4 instance cluster of
    c5.4xlarge, then your `TOTAL_CORE_COUNT` is 32.

3.  Verify that you can run a python command across your cluster. For
    example, run:

        mpiexec -n <TOTAL_CORE_COUNT> -f /shared/.hostfile-<CLUSTER_UUID> python --version

If all commands succeed, you should be able to execute workloads across
your cluster. You can place scripts and small data that are shared
across cluster nodes in `/shared`. However, external storage, such as
S3, should be used for reading and writing large data.

## Running a Job {#running_a_job}

Bodo Cloud Platform has support for running scheduled (and immediate)
Python jobs without the need for Jupyter Notebooks. To create a Job,
navigate to the Jobs page by selecting _Jobs_ in the left bar.

![Sidebar-Jobs](../platform_onboarding_screenshots/side-jobs.png#center)

This pages displays any <inpg> INPROGRESS</inpg> jobs you have previously scheduled
and allows you to schedule new Jobs. At the top right corner, click on
`CREATE JOB`. This opens a job creation form.

First, select a name for your job and specify the cluster on which you
want to deploy your job. If you have an existing cluster that is not
currently bound to a notebook or another job, you can select this
cluster from the dropdown menu. Alternatively, you can create a cluster
specifically for this job by selecting the `NEW` button next to the
cluster dropdown menu. When creating a cluster specifically for a job,
note that the cluster is only used for that job and is removed once the
job completes. After selecting your cluster, indicate when you want your
job to be executed in the **Schedule** section. Then, enter the
**Command** that you want to execute inside this cluster.

!!! note
    This command is automatically prepended with
    `mpiexec -n <CORE_COUNT> python`. For example, to run a file `ex.py`
    with the argument 1, you would enter the command `ex.py 1`.

To specify your source code location, fill in the **Path** line with a
valid Git URL that leads to a repository containing your code.

!!! note
    When selecting a GitHub URL, you should select the URL
    available at the top of your web browser and NOT the path when cloning
    the repository, _i.e._ your path SHOULD NOT end in `.git`.

![Jobs-Forms-Standard](../platform2-gifs/create-job.gif#center)

If you are cloning a private repository, you need to provide the
platform with valid Git credentials to download your repository. To do
so, select `Show advanced` in the bottom right of the form. Then in
**Workspace username**, enter your Git username and in **Workspace
password** enter either your password or a valid Github Access Token.
The advanced options also allow you to specify a particular commit or
branch with **Workspace reference** and to load other custom environment
variables in **Other**.

!!! note
    If your Github Account uses 2FA please use a Github Access
    Token to avoid any possible authentication issues.

Once your form is complete, select `CREATE` to begin your job.

![Job-Run](../platform2-gifs/job-run.gif#center)

Once you've provided all the necessary details, select `CREATE` to
begin your job. You will see a <new>NEW</new> task created in your jobs page.

If you created a cluster specifically for this job, a new cluster will
also appear in your clusters page.

Your job will begin once it reaches its scheduled time and any necessary
clusters have been created. Then your job will transition to being
<inpg>INPROGRESS</inpg>.

At this point your job will execute your desired command. Once it
finishes executing, your job will transition to <fin>FINISHED</fin> status. You
can find any stdout information that you may need by pressing `DETAILS`
followed by `SHOW LOGS`. If a cluster was specifically created for this
job, it will be deleted after the job finishes.

!!! note
    Bodo **DOES NOT** preserve artifacts written to local storage. If
    you have any information that you need to persist and later review, you
    should write to external storage, such as Amazon S3. You may also write
    to stdout/stderr, but output logs may be truncated, so it should not be
    considered reliable for large outputs that need to be read later.

## Troubleshooting

Here are solutions to potential issues you may encounter while using the
Bodo Cloud Platform. 

### Unexpected number of ranks

If you are getting an unexpected number of ranks then the issue could be an inaccurate 
MPI hostfile for the cluster. This is mostly likely to happen after scaling up a cluster. 
You can update the hostfile using IPyParallel Magic `%update_hostfile` and then restart 
the kernel to apply the changes.

```shell
%update_hostfile
```

![Update-Hostfile](../platform2-gifs/update-hostfile.gif#center)

For AWS troubleshooting, refer to this [guide](bodo_platform_aws.md).

For Azure troubleshooting, refer to this [guide](bodo_platform_azure.md).