# Running Batch Jobs
:fontawesome-brands-aws: Supported on AWS ·
:material-microsoft-azure:{.azure} Supported on Azure ·

Bodo supports Python and SQL batch jobs. Batch jobs are useful for running data processing tasks, such as ETL, ELT, data preparation, and data analysis. 
Bodo Cloud Platform allows users to create _Batch Job Definitions_ and then submit _Batch Job Runs_ for execution.

![Sidebar-Jobs](../../platform2-screenshots/side-jobs.png#center)

## Batch Job Definitions 

Batch job definitions are stored objects which can be used to run your data processing and analysis applications in a Bodo Platform workspace.
You need to have an available workspace before creating a batch job definitions. 


![Job-Definition-Form](../../platform2-gifs/workspace-job-def.gif#center)

## Batch Job Definition Form Fields 

Here are the fields you need to provide when creating a batch job definition. 

### Name :fontawesome-solid-asterisk:{.requirement}

The name of the batch job definition. The name must be unique within the workspace. Each batch job definition is assigned a unique id (UUID). 
If a batch job definition is deleted from the workspace, the name can be reused. However, the UUID of the new batch job definition will be different.

### Description

The description of the batch job definition. This is optional. 


### Configuration :fontawesome-solid-asterisk:{.requirement}
A batch job configuration is a JSON object that specifies the batch job's source, environment, and its execution logic. 

It includes the following fields:

- ##### Source :fontawesome-solid-asterisk:{.requirement}

    The source of the batch job.
    
    - ***Type***:fontawesome-solid-asterisk:{.requirement}  You need to specify the job source type. We currently support three types of sources: `WORKSPACE`, `GIT` and `S3`.
    - ***Definition***:fontawesome-solid-asterisk:{.requirement}  You need to specify the source definition. The source definition depends on the source type. 
    
        - **Workspace**: The batch job source is a file in the workspace (`bodofs`). You need to specify the file path in the workspace. 
        - **Git**: The batch job source is a file in a Git repository. You need to specify the Git repository URL and the file path in the repository. 
      You also need to provide a Git *username* and an *access token* for accessing the repository. If you want to check out a specific 
        branch or commit, you can specify the branch or commit `reference`. Otherwise, the default branch will be used.
        - **S3**: The batch job source is a file in an S3 bucket. You need to specify the file path including the bucket name. You also need to provide a bucket region.


- ##### Type :fontawesome-solid-asterisk:{.requirement}
The type of the batch job. Currently, we support two types of batch jobs: Python(`PYTHON`) and SQL(`SQL`). 

- ##### Source Location :fontawesome-solid-asterisk:{.requirement}
The relative path from the location in the job source to the `.py` or `.sql` file that contains the job script.

    !!! note
        For workspace jobs, when you write a job definition, the job will consider the path of the file in the workspace to be the concatenation of the source location and the file path in the source.
        For example, if the source location is `/shared/bodouser` and the file path is `myjob.py`, the path of the file is `/shared/bodouser/myjob.py`. If you don't provide a full path,
        the path of the file will be relative to the workspace working directory. 

- ##### Arguments
The arguments to the batch job. The arguments are passed to the batch job script as command line arguments for `.py` files. 
```json title="Example arguments"
{
    "args": "--arg1 value1 --arg2 value2"
}
```

- ##### Retry Strategy 

    The retry strategy for the batch job. The retry strategy is a JSON object that specifies the retry policy for the batch job. It includes the following fields:

    -  Number of Retries: The number of retries for the batch job. The default value is 0.
    -  Retry Delay: The retry interval in minutes. The default value is one minute.
    -  Auto Retry on Timeout: Whether to retry on job timeout. The default value is `false`.

- ##### Timeout
The timeout for the batch job in minutes. The default value is 60 minutes. 
Note that the timeout applies to each individual retry attempt, and not the total execution time of a batch job run 
with potentially multiple retries. 

- ##### Environment Variables 
Key-value pairs of environment variables for the batch job. Default value is an empty dictionary.


### Cluster Configuration 

The cluster configuration specifies the default cluster configuration for the batch job. This is optional, as you can always choose a cluster to submit the batch job run.

It includes the following fields:

- ##### Cluster Instance Type :fontawesome-solid-asterisk:{.requirement}
The cluster instance type depending on the cloud service provider. 

- ##### Workers Quantity :fontawesome-solid-asterisk:{.requirement}
The number of workers in the cluster.

- ##### Bodo Version :fontawesome-solid-asterisk:{.requirement}
The Bodo version to use for the cluster.

## Running a Batch Job
Once you've created a batch job definition, you can run it whenever you want. 

![Job-Run](../../platform2-gifs/job-run.gif#center)

### Batch Job Run Form Fields
To submit a batch job run, you need to provide the following information:

- ##### Batch Job ID 
The ID of the batch job to run.

- ##### Cluster UUID 
    
    The UUID of the cluster to run the batch job on. 
    
    !!! note
        - If you don't provide a cluster UUID, the batch job will run on a new cluster with the default cluster configuration provided by the associated definition.
        - If neither a cluster UUID nor a cluster configuration is provided, an error will be thrown.


### Batch Job Run Logs and Status

Batch jobs can have one of the following statuses. 

- `PENDING` - The batch job is pending.
- `RUNNING` - The batch job is running.
- `SUCCEEDED` - The batch job run succeeded.
- `FAILED` - The batch job run failed.
- `CANCELLED` - The batch job run was cancelled.

Each batch job that is not `SUCCEEDED` also has a status reason associated with it. When the status is one of `PENDING`, `FAILED` or `CANCELLED`, the reason could be one of the following:

- `Cancelled by user` - The batch job run was `CANCELLED` by the user.
- `In queue` - The batch job is `PENDING` in the queue because there's potentially another batch job running on the same cluster. 
- `Recently submitted` - The batch job is `PENDING` because it was recently submitted. 
- `NonZeroExitCode` - The batch job run `FAILED` because the job script exited with a non-zero exit code.
- `Timeout` - The batch job run `FAILED` because it timed out.

You can access the logs of a batch job run from the UI as well.

![Job-Run](../../platform2-gifs/job-run-logs.gif#center)


### Running a SQL Query as a Batch Job {#sql-batch-job}

Bodo supports running SQL queries as batch jobs without explicitly writing a batch job definition. 
See [Bodo Platform SDK][submit-sql-job-run] for usage details.

### Queuing Batch Job Runs

If you submit a batch job run while there's another batch job running on the same cluster, 
the new batch job will automatically be queued. Currently, at most 100 job runs can be queued on a cluster at a time.
Note that you can queue job runs for different batch job definitions on the same cluster.

### Submitting Batch Job Runs to a Paused Cluster

You can choose whether to allow a cluster to resume on submission of a job run. This is enabled by default. 

If you submit a batch job run to a paused cluster with auto-resume enabled,
the cluster will be resumed automatically. If auto-resume is not enabled, then the job run submission will fail. 


