# Using Bodo Cloud Platform {#bodo_platform}

## Bodo Cloud Platform Concepts {#bodo_platform_concepts}

This page describes the fundamental concepts needed to use the Bodo Cloud Platform.

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
   This allows the platform to provision and manage cloud resources in your account.
2. Details regarding metadata storage. The platform needs to store
   specific metadata to carry out its functions, such as the state of your various cloud deployments, logs, etc.
   On AWS, this data is stored in an S3 bucket and a DynamoDB table.
   On Azure, this data is stored in a storage container.


![Cloud-Configs](../platform2-screenshots/cloud_configs.png#center)

### Workspaces

A *workspace* on the Bodo Cloud Platform consists of:

1. A shared filesystem where you can collaborate with your team on your projects.
2. Networking infrastructure such as virtual networks, security groups, and subnets in which
   your compute clusters and Jupyter servers will be securely deployed.

A workspace is tied to a particular cloud configuration and has its own user management, i.e.,
you can have different subsets of users with different sets of roles
and permissions in different workspaces within the same organization.

!!! info "Important"
     If a user who is not part of the organization is invited to a workspace in the organization,
    they are automatically added to the organization with minimal permissions.

![Workspaces](../platform2-screenshots/workspace.png#center)

To create a workspace, go to the "Workspaces" section in the sidebar and click "Create Workspace." In the
creation form, enter the name of the workspace, select the cloud configuration to use for provisioning
it and the region where it should be deployed, and click on "Create Workspace."

![Create-Workspace](../platform2-screenshots/create-workspace-form.png#center)

This will start the workspace deployment. When the workspace is in the "READY" state, click on
the button next to it to enter it.

![Enter-Workspace](../platform2-screenshots/enter-workspace.png#center)

## Notebooks

Jupyter servers act as your interface to your shared file system and compute clusters.
Users can execute code from their notebooks on the compute cluster from the Jupyter interface.
A Jupyter server is automatically provisioned when you first enter the workspace.

![Notebook-View](../platform2-screenshots/notebook_view.png#center)

You can update/restart Jupyter servers in the "Workspace Settings."


![Notebook-Manager](../platform2-gifs/notebook_manager.gif#center)

## Clusters {#creating_clusters}

In the left bar, click on _Clusters_ (or click on the second step in the
_Onboarding_ list). This action will take you to the _Clusters_ page. At the top right corner,
click `Create Cluster,` which opens the cluster creation form.

![Cluster-Create](../platform2-gifs/create_cluster.gif#center)

### Cluster basic configuration

![Cluster-Form](../platform2-gifs/create_cluster_form.gif#center)

**Cluster Name:** 

Choose a name for your cluster.

**Instance type:** 

Select the type of nodes in the cluster to be created from the dropdown list.
[EFA](https://aws.amazon.com/hpc/efa/){target="blank"} will be used if the instance type supports it.

![Cluster-Form-Instance](../platform2-gifs/create_cluster_list.gif#center)

!!! note
    If the **Instance type** dropdown list does not populate,
    either the credentials are not entered properly or they are not valid.
    Please see how to set your [AWS][setting_aws_credentials]
    or [Azure][setting_azure_credentials] credentials and make sure your credentials are valid.

**Use Spot Instances:** 

This option enables spot instances in the cluster. Use this option to reduce the cost of VMs.

![Cluster-Spot-Instance](../platform2-screenshots/use_spot_instance.png#center)

!!! note
    However, it's important to note that selecting this option can also have some drawbacks. 
    For further insights, please refer to the breakdowns associated with AWS and Azure spot instances.
    [Azure Spot](https://azure.microsoft.com/en-us/products/virtual-machines/spot), [AWS Spot](https://aws.amazon.com/ec2/spot/)

**Number of Instances:** 

This option specifies the number of nodes in your cluster.

**Bodo Version:** 

This option specifies the Bodo version to be installed on your cluster. 
Typically, the three latest Bodo Releases are available.

![Cluster-Form-Bodo](../platform2-screenshots/cluster_bodo_version.png#center)

**Cluster Auto Pause:** 

This is the amount of time of inactivity after which the platform will pause the cluster automatically.

![Cluster-Form-Auto-Pause](../platform2-screenshots/cluster_auto_pause.png#center)

### Cluster advanced configuration
Additionally, you can specify the following advanced configuration options for cluster.

![Cluster-Form-Advanced](../platform2-screenshots/cluster_advanced.png#center)

**Availability Zone:** 

:fontawesome-brands-aws: On AWS only

Select the availability zone where you want to deploy your cluster. By default, this is set to `Auto Select`.

**Auto Stop:** 

Activity is determined through attached notebooks (see
[how to attach a notebook to a cluster][attaching_notebook_to_cluster]) and jobs
(see [how to run a job][running-a-batch-job]). Therefore, if you
don't plan to attach a notebook or a job to this cluster (and use it
via `ssh` instead), we recommend that you to set this to
`Never,` since otherwise, the cluster will be stopped after
the set time.

**Instance Role:** 

:fontawesome-brands-aws: On AWS only 

Is the instance role that should be attached to the cluster instances. 
You can define these in Settings. By default, a new role will be created and attached.

**Cluster description:** 

Description for the cluster.

Finally click on `CREATE`. You will see that a new task for creating the
cluster has been created. The status is updated to <inpg>INPROGRESS</inpg> when the task starts executing and
cluster creation is in progress.

![Cluster-Status-InProgress](../platform2-screenshots/cluster_inprogress.png#center)

You can click on the `Details` drop-down to monitor the progress of the
cluster creation.

![Cluster-Info](../platform2-screenshots/cluster_inprogress_deatails.png#center)

Once the cluster is successfully created and ready to use, the status is
updated to <fin>RUNNING</fin>.

![Cluster-Status-Finished](../platform2-gifs/create_cluster_details.gif#center)

### Cluster Instance Type and Size Recommendations
If you were previously running a query on a Snowflake Warehouse this table provides a starting point for what
instance type and size you can use to run the query using Bodo. Since this is only a starting point you should
experiment to find the best configuration for your specific use case.

| Snowflake Warehouse Size | Bodo Cluster Spec |
|--------------------------|-------------------|
| 2X-Small                 | 1 x i4i.xlarge    |
| X-Small                  | 1 x i4i.2xlarge   |
| Small                    | 1 x i4i.4xlarge   |
| Medium                   | 1 x i4i.8xlarge   |
| Large                    | 1 x i4i.16xlarge  |
| X-Large                  | 1 x i4i.32xlarge  |
| 2X-Large                 | 2 x i4i.32xlarge  |
| 3X-Large                 | 4 x i4i.32xlarge  |
| 4X-Large                 | 8 x i4i.32xlarge  |
| 5X-Large                 | 16 x i4i.32xlarge |
| 6X-Large                 | 32 x i4i.32xlarge |


## Attaching a Notebook to a Cluster {#attaching_notebook_to_cluster}

To attach a notebook to a cluster, select the cluster from the drop-down in the top-left.

![Attach-Cluster](../platform2-gifs/attach-cluster.gif#center)

To execute your code across the attached cluster, select the _Parallel Python_ cell type from the cell type selector dropdown.

![Run-Code-Notebook](../platform2-gifs/parallel-python.gif#center)

To run a SQL query, first select the catalog you want to use, then select the SQL cell type from the cell type selector dropdown. 
For more information on SQL catalogs, refer to the [SQL Catalogs usage guide][sql_catalog].

![Run-Code-Notebook](../platform2-gifs/sql.gif#center)


!!! note 
    Execution is only allowed when the notebook is attached to a cluster.
    If you execute a cell without a cluster attached, the following warning will be shown:

![Detached-Notebook-Warning](../platform2-gifs/not-attached-to-cluster-warning.gif#center)

## Using your own Instance Role for a Cluster {#instance_role_cluster}
### AWS
In cases where you want to access additional AWS resources from Bodo clusters e.g., S3 buckets,
you can create an IAM Role in your AWS account and then register it as an Instance Role on the Bodo Platform, which will allow you to access those resources from Bodo clusters without using AWS keys.

![View-Instance-Roles](../platform2-gifs/instance-role-list.gif#center)

Note that, by default, Bodo creates an IAM role with the necessary policies for each cluster. When you register your own role with the Bodo Platform, it will automatically attach the other required policies to this role.

Here, we walk through setting up an IAM Role in AWS and then registering it as an Instance Role on the Bodo Platform. For this example, we will be creating a role with access to an S3 bucket in your AWS account:

Step 1: Create an AWS IAM Role on the [AWS Management Console](https://aws.amazon.com/console/):
1. Go to the IAM service.

![AWS-IAM](../platform2-screenshots/aws_iam.png#center)

2. In the left sidebar click on Roles.

![AWS-IAM-ROLE](../platform2-screenshots/aws_iam_role.png#center)

3. Click on the button `Create role`, then select:
   * Trusted entity type: **AWS service**
   * Common use cases: **EC2**

![AWS-IAM-Role-Form](../platform2-screenshots/aws_iam_role_form.png#center)

4. Click next, and then create a new policy that will be attached to this role:
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

5. Go back to Create role, refresh the list of policies, and add the policy that you created.
6. Click **Next**, then in the Role name field, type a role name and click **Create role**.
7. Copy the Role ARN from the role summary.

![AWS-IAM-Role-ARN](../platform2-screenshots/aws_iam_role_arn.png#center)


Step 2: Register your AWS IAM Role on the Bodo Platform as a new Instance Role:

1. Click on the **CREATE INSTANCE ROLE** button and in the creation form, fill the following fields:
   * Name: Name for the Instance Role 
   * Role ARN: AWS Role ARN from Step 1
   * Description: Short description for Instance Role

![Create-Instance-Role](../platform2-gifs/instance-role-form.gif#center)


2. Click on the **Create** button.


The Instance Role will now be registered on the Platform. It can have one of two status-es:
* **Active**: Instance Role is ready to use
* **Failed**: Something went wrong while registering the Instance Role and it cannot be used. Some possible problems could be:
   * The Platform wasn't able to find the specified role.
   * The Platform was not able to attach additional Bodo polices that are required for normal cluster operations.

## Managing Packages on the cluster using IPyParallel magics - Conda and Pip

We recommend all packages to be installed using Conda as that is what we use in our environments.
Any conda command can be run in parallel on all the nodes of your cluster using `%pconda`.
To install a new package on all the nodes of your cluster you can use `%pconda install`.
All conda install arguments work as expected, e.g., `-c conda-forge` to set the channel.

```shell
%pconda install -c conda-forge <PACKAGE_NAME>
```

To learn more about the packages installed on the cluster nodes `%pconda list`.
```shell
%pconda list
```

To remove a conda package on all your cluster nodes, use `%pconda remove`.

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
Notebooks and Jobs. However, it may be necessary to connect directly to
a cluster in some cases. In that case, you can connect through a notebook
terminal.

### Connecting with a Notebook Terminal
![Notebook-Terminal](../platform2-gifs/notebook-terminal.gif#center)

If your cluster has more than one node , you can connect to any of the cluster nodes by running the following command in the terminal:

```shell
ssh <NODE-IP>
```

![Connect-Cluster](../platform2-gifs/connect-to-cluster.gif#center)

Through this terminal, you can access the `/bodofs` folder, which
is shared by all the instances in the cluster and the Notebook instance.
[Verify your connection][verify_your_connection] to interact directly with your cluster.

### Verify your Connection {#verify_your_connection}

Once you have connected to a node in your cluster, you should verify
that you can run operations across all the instances in the cluster.

1. Verify the path to the hostfile for your cluster. You can find it by
    running:
    ```shell
    ls -la /home/bodo/hostfile
    ```
2. Check that you can run a command across you cluster. To do this,
    run:

    ```shell
    mpiexec -n <TOTAL_CORE_COUNT> -f /home/bodo/hostfile hostname
    ```

    This will print one line per each core in the cluster, with one
    unique hostname per cluster node.

    Your cluster's `TOTAL_CORE_COUNT` is usually half the
    number of vCPUs on each instance times the number of instances in
    your cluster. For example, if you have a 4 instance cluster of
    c5.4xlarge, then your `TOTAL_CORE_COUNT` is 32.

3. Verify that you can run a python command across your cluster. For
    example, run:

        mpiexec -n <TOTAL_CORE_COUNT> -f /home/bodo/hostfile python --version

If all commands succeed, you should be able to execute workloads across
your cluster. You can place scripts and data that are shared
across cluster nodes in `/bodofs`. 

## Running a Batch Job
Bodo Cloud Platform has support for running Python and SQL batch jobs.

![Sidebar-Jobs](../platform2-screenshots/side-jobs.png#center)

1. Creating batch job definition:

    ![Job-Definition-Form](../platform2-gifs/workspace-job-def.gif#center)


2. Running a batch job:

    ![Job-Run](../platform2-gifs/job-run.gif#center)


3. Job run logs:

    ![Job-Run](../platform2-gifs/job-run-logs.gif#center)

For further information, please refer to the [Batch Job Guide.][batch-jobs]

## Troubleshooting

Here are solutions to potential issues you may encounter while using the
Bodo Cloud Platform. 

### Notebook: 403 Forbidden

![Notebook-403-Error](../platform2-screenshots/notebook_403_error.png#center)

This error is typically caused by an incorrect token, often occurring when a token is cached from accessing a different workspace. 
There are two recommended workarounds:

First, attempt to resolve the issue by navigating back to the organization using the left menu `Back to Organization`. 
Once there, re-enter the workspace.

If the problem persists, try clearing your web browser's cache and then logging in to Bodo Platform once more.

### Notebook: 404 Not Found / 502 Bad Gateway

![Notebook-404-Error](../platform2-screenshots/notebook_404_error.png#center)

If for some reason `My Notebook` is displaying error code 404 / 502, you should try to update and then restart the notebook server
using left menu. If this does not help, please [contact us](https://bodo.ai/contact/){target="blank"} for further assistance.

### Notebook: File Save Error

![Notebook-File-Save-Error](../platform2-screenshots/file_save_error.png#center)

If you get a file save error with message `invalid response: 413`
make sure your notebook (`.ipynb`) file is less than 16MB in size. Bodo Platform
does not support notebook files larger than 16MB in size.
To reduce file size don't print large sections of text and clear output
cells by clicking `Edit` > `Clear All Outputs` in the notebook interface.

### Account Locked Error

![Account-Locked-Error](../platform2-screenshots/account_locked.png#center)

When you login to the platform, if you get an account locked error with message `User is locked out. To unlock user, please contact your administrators`,
this means that your account has been dormant (no login in more than 90 days). Please [contact us](https://bodo.ai/contact/){target="blank"} to unlock your account.

For AWS troubleshooting, refer to this [guide](bodo_platform_aws.md).

For Azure troubleshooting, refer to this [guide](bodo_platform_azure.md).
