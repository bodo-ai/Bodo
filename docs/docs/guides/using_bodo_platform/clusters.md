# Creating a Cluster {#creating_clusters}

Clusters are the compute resources that you can use to run your Bodo code in your workspace. This guide explains how to create a cluster on the Bodo Cloud Platform.
In the left bar, click on _Clusters_ (or click on the second step in the
_Onboarding_ list). This action will take you to the _Clusters_ page. At the top right corner,
click `Create Cluster,` which opens the cluster creation form.

![Cluster-Create](../../platform2-gifs/create_cluster.gif#center)

### Cluster Basic Configuration

![Cluster-Form](../../platform2-gifs/create_cluster_form.gif#center)

#### Cluster Name

Choose a name for your cluster.

#### Instance type

Select the type of nodes in the cluster to be created from the dropdown list.
[EFA](https://aws.amazon.com/hpc/efa/){target="blank"} will be used if the instance type supports it.

![Cluster-Form-Instance](../../platform2-gifs/create_cluster_list.gif#center)

!!! note
If the **Instance type** dropdown list does not populate,
either the credentials are not entered properly or they are not valid.
Please see how to set your [AWS][setting_aws_credentials]
or [Azure][setting_azure_credentials] credentials and make sure your credentials are valid.

#### Use Spot Instances

This option enables spot instances in the cluster. Use this option to reduce the cost of VMs.

![Cluster-Spot-Instance](../../platform2-screenshots/use_spot_instance.png#center)

!!! note
However, it's important to note that selecting this option can also have some drawbacks.
For further insights, please refer to the breakdowns associated with AWS and Azure spot instances.
[Azure Spot](https://azure.microsoft.com/en-us/products/virtual-machines/spot), [AWS Spot](https://aws.amazon.com/ec2/spot/)

#### Number of Instances

This option specifies the number of nodes in your cluster.

#### Bodo Version

This option specifies the Bodo version to be installed on your cluster.
Typically, the three latest Bodo Releases are available.

![Cluster-Form-Bodo](../../platform2-screenshots/cluster_bodo_version.png#center)

#### Cluster Auto Pause

This is the amount of time of inactivity after which the platform will pause the cluster automatically.

![Cluster-Form-Auto-Pause](../../platform2-screenshots/cluster_auto_pause.png#center)

### Cluster Advanced Configuration

Additionally, you can specify the following advanced configuration options for cluster.

![Cluster-Form-Advanced](../../platform2-screenshots/cluster_advanced.png#center)

#### Availability Zone

:fontawesome-brands-aws: On AWS only

Select the availability zone where you want to deploy your cluster. By default, this is set to `Auto Select`.

#### Auto Stop

Activity is determined through attached notebooks (see
[how to attach a notebook to a cluster][attaching_notebook_to_cluster]) and jobs
(see [how to run a job][running-a-batch-job]). Therefore, if you
don't plan to attach a notebook or a job to this cluster (and use it
via `ssh` instead), we recommend that you to set this to
`Never,` since otherwise, the cluster will be stopped after
the set time.

#### Instance Role

:fontawesome-brands-aws: On AWS only

Is the instance role that should be attached to the cluster instances.
You can define these in Settings. By default, a new role will be created and attached.

#### Cluster description

Description for the cluster.

### Cluster Instance Type and Size Recommendations

If you were previously running a query on a Snowflake Warehouse this table provides a starting point for what
instance type and size you can use to run the query using Bodo. Since this is only a starting point you should
experiment to find the best configuration for your specific use case.

| Snowflake Warehouse Size | Bodo Cluster Spec |
|--------------------------|-------------------|
| 2X-Small | 1 x i4i.xlarge |
| X-Small | 1 x i4i.2xlarge |
| Small | 1 x i4i.4xlarge |
| Medium | 1 x i4i.8xlarge |
| Large | 1 x i4i.16xlarge |
| X-Large | 1 x i4i.32xlarge |
| 2X-Large | 2 x i4i.32xlarge |
| 3X-Large | 4 x i4i.32xlarge |
| 4X-Large | 8 x i4i.32xlarge |
| 5X-Large | 16 x i4i.32xlarge |
| 6X-Large | 32 x i4i.32xlarge |

### Monitoring the Cluster Creation Status

Once you have filled in all the required click on `CREATE`. You will see that a new task for creating the
cluster has been created. The status is updated to <inpg>INPROGRESS</inpg> when the task starts executing and
cluster creation is in progress.

![Cluster-Status-InProgress](../../platform2-screenshots/cluster_inprogress.png#center)

You can click on the `Details` drop-down to monitor the progress of the
cluster creation.

![Cluster-Info](../../platform2-screenshots/cluster_inprogress_deatails.png#center)

Once the cluster is successfully created and ready to use, the status is
updated to <fin>RUNNING</fin>.

![Cluster-Status-Finished](../../platform2-gifs/create_cluster_details.gif#center)
