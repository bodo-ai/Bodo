# Running Jobs
:fontawesome-brands-aws: Supported on AWS · :material-microsoft-azure:{.azure} Supported on Azure ·

Bodo supports Python and SQL jobs. Jobs are useful for running data processing tasks, such as ETL, ELT, data preparation, and data analysis. 
Bodo Cloud Platform allows users to create _Job Template_ and then submit _Job Runs_ for execution.

![Sidebar-Jobs](../../platform2-screenshots/side-jobs.png#center)

## Job Template 

Job templates are stored objects which can be used to run your data processing and analysis applications in a Bodo Platform workspace.
You need to have an available workspace before creating a job templates. 


![Job-Template-Form](../../platform2-gifs/workspace-job-def.gif#center)

## Job Template Form Fields 

Here are the fields you need to provide when creating a job template. 

### 1) Basic: :fontawesome-solid-asterisk:{.requirement}

- ***Name***: :fontawesome-solid-asterisk:{.requirement} The job template name.

- ***Type***: :fontawesome-solid-asterisk:{.requirement} We currently support three types of sources: `WORKSPACE`, `GIT`, `URL` and `S3`.

    !!! note 
        The job template name must be unique within the workspace. Each job template is assigned a unique id (UUID). 
        If a job template is deleted from the workspace, the name can be reused. However, the UUID of the new job template will be different.


### 2) Source: :fontawesome-solid-asterisk:{.requirement}

- ***Type*** :fontawesome-solid-asterisk:{.requirement}:
The type of the job. Currently, we support three types of jobs: Python(`PYTHON`) SQL(`SQL`), Notebook(`IPYNB`).

- ***File Name*** :fontawesome-solid-asterisk:{.requirement}:
The relative path from the job source to the `.py`, `.sql` and `ipynb` file that contains the job script.

- ***Timeout***: The timeout for the job in minutes. The default value is 60 minutes. 
Note that the timeout applies to each individual retry attempt, and not the total execution time of a job run with potentially multiple retries.

##### Workspace
The job source is a file in the workspace (`bodofs`). You need to specify the path and file name in the workspace.
  ![Job-Workspace-Source](../../platform2-screenshots/workspace-source.png#center)

##### GIT
The job source is a file in a Git repository. You need to specify the Git repository URL and the file path in the repository.
You also need to provide a Git *username* and an *access token* for accessing the repository. If you want to check out a specific
branch or commit, you can specify the branch or commit `reference`. Otherwise, the default branch will be used.
  ![Job-Git-Source](../../platform2-screenshots/git-source.png#center)

##### URL
The job source is a file under specific url. You need to specify the base url and file name.
  ![Job-Url-Source](../../platform2-screenshots/url-source.png#center)

##### S3
The job source is a file in an S3 bucket. You need to specify the file path including the bucket name. You also need to provide a bucket region.
  ![Job-S3-Source](../../platform2-screenshots/s3-source.png#center)


### 3) Advanced Options
![Job-Advanced-Options](../../platform2-screenshots/advanced-options.png#center)

- ##### Arguments
    The arguments to the job. The arguments are passed to the job script as command line arguments for `.py` files. 
    ```json title="Example arguments"
    {
        "args": "--arg1 value1 --arg2 value2"
    }
    ```

- ##### Environment Variables
    Key-value pairs of environment variables for the job. Default value is an empty dictionary.

- ##### Catalog
    Catalog is configuration object that grant BodoSQL access to load tables from a database.

- ##### Retry Strategy 

    The retry strategy for the job. The retry strategy is a JSON object that specifies the retry policy for the job. It includes the following fields:

    -  Number of Retries: The number of retries for the job. The default value is 0.
    -  Retry Delay: The retry interval in minutes. The default value is one minute.
    -  Auto Retry on Timeout: Whether to retry on job timeout. The default value is `false`.

- ##### Description
The description of the job template.

### 4) Job Cluster 

The job cluster configuration specifies the default cluster configuration for the job. 
This is optional, as you can always choose a cluster to submit the job run.
It includes the following fields:

- ##### Cluster Name :fontawesome-solid-asterisk:{.requirement}
    The cluster name that will be created.

- ##### Instance Type :fontawesome-solid-asterisk:{.requirement}
    The cluster instance type depending on the cloud service provider. 

- ##### Number of Workers :fontawesome-solid-asterisk:{.requirement}
    The number of workers in the cluster.

- ##### Bodo Version :fontawesome-solid-asterisk:{.requirement}
    The Bodo version to use for the cluster.


## Running a Job
Once you've created a job template, you can run it whenever you want.

![Job-Run](../../platform2-gifs/job-run.gif#center)

### Create Job Run Form
To submit a new job from template, you need to select:

- ##### Create New Cluster from Job Template
    Option only enabled if in job template definition we have defined a cluster default configuration.

- ##### Pause Cluster on finish
    Pause a cluster after job finish.

- ##### Cluster UUID
    The UUID of the cluster to run the job on. 
    
!!! note
    - If you don't provide a cluster UUID, the job will run on a new cluster with the default cluster configuration provided by the associated definition.
    - If neither a cluster UUID nor a cluster configuration is provided, an error will be thrown.


### Job Logs and Status

Jobs can have one of the following statuses. 

- `PENDING` - The job is pending.
- `RUNNING` - The job is running.
- `SUCCEEDED` - The job run succeeded.
- `FAILED` - The job run failed.
- `CANCELLED` - The job run was cancelled.

Each job that is not `SUCCEEDED` also has a status reason associated with it. When the status is one of `PENDING`, `FAILED` or `CANCELLED`, the reason could be one of the following:

- `Cancelled by user` - The job run was `CANCELLED` by the user.
- `In queue` - The job is `PENDING` in the queue because there's potentially another job running on the same cluster. 
- `Recently submitted` - The job is `PENDING` because it was recently submitted. 
- `NonZeroExitCode` - The job run `FAILED` because the job script exited with a non-zero exit code.
- `Timeout` - The job run `FAILED` because it timed out.

You can access the logs of a job run from the UI as well.

![Job-Run](../../platform2-gifs/job-run-logs.gif#center)


### Running a SQL Query as a Job {#sql-job}

Bodo supports running SQL queries as jobs without explicitly writing a job template. 
See [Bodo Platform SDK][execute-sql-query] for usage details.

### Queuing Job Runs

If you submit a job run while there's another job running on the same cluster, 
the new job will automatically be queued. Currently, at most 100 job runs can be queued on a cluster at a time.
Note that you can queue job runs for different job templates on the same cluster.

### Submitting Job to a Paused Cluster

You can choose whether to allow a cluster to resume on submission of a job run. This is enabled by default. If you submit a job to a paused cluster with auto-resume enabled,
the cluster will be resumed automatically. If auto-resume is not enabled, then the job run submission will fail. 


