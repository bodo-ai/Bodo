# Using Bodo Cloud Platform {#bodo_platform}

## Creating Clusters {#creating_clusters}

In the left bar click on *Clusters* (or click on the second step in the
*Onboarding* list):

![Sidebar-Clusters](../platform_onboarding_screenshots/side-clusters.png#center#center)

This will take you to the *Clusters* page. At the top right corner,
click on `Create Cluster` which opens the cluster creation form. First,
choose a name for your cluster and check the
[EFA](https://aws.amazon.com/hpc/efa/){target="blank"} checkbox if you want to use
EFA-enabled nodes (only available on AWS). Then, select the type of
nodes in the cluster to be created from the **Instance type** dropdown
list.

!!! note
     If the **Instance type** dropdown list does not populate,
    either the credentials are not entered properly or they are not valid.
    Please see how to set your [AWS][setting_aws_credentials]
    or [Azure][setting_azure_credentials] credentials and make sure your credentials are valid.

Next, enter the number of nodes for your cluster in **Number of
Instances**. and choose the Bodo Version to be installed on your
cluster. Typically the three latest Bodo Releases are available.

!!! note 
    If the **Bodo Version** dropdown list does not populate,
    either the credentials are not entered properly or the permissions to
    Bodo's Images have not been granted to your account. Please repeat the steps
    to set your [AWS][setting_aws_credentials] or [Azure][setting_azure_credentials]
    credentials and make sure you complete themwith valid credentials and that images have been
    successfully shared with your AWS or Azure account.

Then, select a value for **Cluster auto shutdown**. This is the amount
of time of inactivity after which the platform will remove the cluster
automatically. Activity is determined through attached notebooks (see
[how to attach a notebook to a cluster][attaching_notebook_to_cluster]) and jobs
(see [how to run a job][running_a_job]). Therefore, if you
don't plan to attach a notebook or a job to this cluster (and use it
via `ssh` instead), it's recommended to set this to
`Never`, since otherwise the cluster will be removed after
the set time.

![Cluster-creation-form](../platform_onboarding_screenshots/cluster-form.png#center#center)

Finally click on `CREATE`. You will see that a new task for creating the
cluster has been created.

![Cluster-Status-New](../platform_onboarding_screenshots/cluster-status-new.png#center#center)

The status is updated to *INPROGRESS* when the task starts executing and
cluster creation is in progress.

![Cluster-Status-InProgress](../platform_onboarding_screenshots/cluster-status-ip.png#center#center)

You can click on the `Details` drop down to monitor the progress for the
cluster creation.

![Cluster-Info](../platform_onboarding_screenshots/cluster-info.png#center#center)

Once the cluster is successfully created and ready to use, the status is
updated to *FINISHED*.

![Cluster-Status-Finished](../platform_onboarding_screenshots/cluster-status-done.png#center#center)

## Attaching a Notebook to a Cluster {#attaching_notebook_to_cluster}

Go to the notebooks page by clicking on *Notebooks* in the left bar (or
on the third green step in the *Onboarding* list at the top).

![Sidebar-Notebooks](../platform_onboarding_screenshots/side-nbs.png#center)

This will take you to the *Notebooks* page. At the top right corner,
click on the `Create Notebook` button which opens the notebook creation
form. Choose a name for your notebook and select the type of node that
will host the notebook from the **Instance type** drop down list. Note
that this node is for running the Jupyter notebook itself, and will not
run cluster workloads. Lastly, select a cluster for attaching the
notebook from the **Cluster** drop down menu and and click on `CREATE`.

![Notebook-Creation-Form](../platform_onboarding_screenshots/nb-form.png#center)

After clicking `CREATE`, a new task for creating the notebook and its
corresponding node is created.

![Notebook-Status-New](../platform_onboarding_screenshots/nb-status-new.png#center)

The status updates to *INPROGRESS* when the task starts executing.

![Notebook-Status-InProgress](../platform_onboarding_screenshots/nb-status-ip.png#center)

After creating the notebook, the platform runs readiness probe checks:

![Notebook-Status-ReadinessProbe](../platform_onboarding_screenshots/nb-status-rp.png#center)

The notebook is ready to use after all checks are complete.
`OPEN NOTEBOOK` will open the notebook in the current browser page,
while the dropdown allows opening the notebook in a new tab.

![Notebook-Status-Finished](../platform_onboarding_screenshots/nb-status-done.png#center)

## Connecting to a Cluster {#connecting_to_a_cluster}

We recommend interacting with clusters primarily through Jupyter
notebooks and Jobs. However, it may be necessary to connect directly to
a cluster in some cases. You can either connect through a notebook
terminal (recommended), or ssh directly from your machine. The latter
requires providing your ssh public key during cluster creation.

### Connecting with a Notebook Terminal

First, you need to [create a cluster][creating_clusters]
and [attach a notebook to the cluster][attaching_notebook_to_cluster]. 

Then, go the cluster tab and find your cluster. Click on `DETAILS` and
copy the cluster `UUID`.

![Cluster-UUID-Info](../platform_onboarding_screenshots/cluster-ip-info.png#center)

Next, go to the notebooks tab and select `OPEN NOTEBOOK`. In the
*Launcher*, click on `Terminal`.

![Notebook-Terminal](../platform_onboarding_screenshots/notebook-terminal.png#center)

Through this terminal, you can interact with the `/shared` folder, which
is shared by all the instances in the cluster and the Notebook instance.
[Verify your connection][verify_your_connection] to interact directly with your cluster.

### SSH From Your Machine

First, navigate to the clusters tabs and select `Create a Cluster`.
Click on `Show Advanced` and add your public key in **SSH Public Key**.
Then, click on `Add your IP` in the **Access from IP address** section
to enable accessing your cluster from your machine.

![Cluster-Creation-Advanced-Settings](../platform_onboarding_screenshots/cluster-create-advanced-settings.png#center)

Fill the rest of the form by following the steps to [create clusters][creating_clusters].

In the clusters tab, select your cluster and click on `DETAILS` to find
the list of IP addresses for your cluster nodes. Use any of the IP
addresses as the ssh destination. In addition, also copy the cluster
UUID which will be needed to execute commands across the cluster.

![Cluster-IP-Info](../platform_onboarding_screenshots/cluster-ip-info.png#center)

In any ssh agent, you can connect to one of your nodes with:

```shell
ssh -i <path_to_private_key> bodo@<IP_ADDRESS>
```

To add additional ssh options please refer to the documentation for your
ssh agent.

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
navigate to the Jobs page by selecting *Jobs* in the left bar.

![Sidebar-Jobs](../platform_onboarding_screenshots/side-jobs.png#center)

This pages displays any *INPROGRESS* jobs you have previously scheduled
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
valid Git URL or S3 URI (only available on AWS) that leads to a
repository containing your code.

!!! note
     When selecting a GitHub URL, you should select the URL
    available at the top of your web browser and NOT the path when cloning
    the repository, *i.e.* your path SHOULD NOT end in `.git`.
    If selecting an S3 URI, your S3 bucket must be in the same region as
    your cluster.

![Jobs-Forms-Standard](../platform_onboarding_screenshots/jobs-form-standard.png#center)

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

![Jobs-Forms-Advanced](../platform_onboarding_screenshots/jobs-form-advanced.png#center)

Once you've provided all the necessary details, select `CREATE` to
begin your job. You will see a *NEW* task created in your jobs page.

![New-Job](../platform_onboarding_screenshots/jobs-new.png#center)

If you created a cluster specifically for this job, a new cluster will
also appear in your clusters page.

![New-Job-Cluster](../platform_onboarding_screenshots/jobs-cluster-inprogress.png#center)

Your job will begin once it reaches its scheduled time and any necessary
clusters have been created. Then your job will transition to being
*INPROGRESS*.

![InProgress-Job](../platform_onboarding_screenshots/jobs-inprogress.png#center)

At this point your job will execute your desired command. Once it
finishes executing, your job will transition to *FINISHED* status. You
can find any stdout information that you may need by pressing `DETAILS`
followed by `SHOW LOGS`. If a cluster was specifically created for this
job, it will be deleted after the job finishes.

![Finished-Job](../platform_onboarding_screenshots/jobs-finished.png#center)

!!! note
     Bodo DOES NOT preserve artifacts written to local storage. If
    you have any information that you need to persist and later review, you
    should write to external storage, such as Amazon S3. You may also write
    to stdout/stderr, but output logs may be truncated, so it should not be
    considered reliable for large outputs that need to be read later.
