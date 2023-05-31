# Bodo Platform SDK

Bodo Cloud Platform provides a simple SDK that can be integrated
in CI/CD pipelines easily.
For example, compute jobs can be orchestrated
easily.

## Getting started {#getting-started}

Install the latest Bodo SDK using:

```console
pip install bodosdk
```

The first step is to create an *API Token* in the Bodo Platform for
Bodo SDK authentication. Navigate to *API Tokens* in the Admin Console to generate a token.
Copy and save the token's *Client ID* and *Secret Key* and use them for BodoClient
definition:

```python
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
```

Alternatively, set `BODO_CLIENT_ID` and `BODO_SECRET_KEY` environment variables
to avoid requiring keys:

```python
from bodosdk.client import get_bodo_client

client = get_bodo_client()
```

**Other bodo client options**

- `print_logs` - default `False`, if enabled all API calls will be printed

```python
from bodosdk.client import get_bodo_client
from bodosdk.models import WorkspaceKeys

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys, print_logs=True)
```

## Job resource {#job-resource}

Module responsible for managing jobs in workspace.

### Bodo Platform Jobs :material-delete-clock:{ title="Soon to be deprecated"}


<br/>
##### Create job   

`BodoClient.job.create(job: JobDefinition)`  

Creates a job to be executed on cluster. You can either create job dedicated cluster by providing its definition or
provide existing cluster uuid. Job dedicated clusters will be removed as soon as job execution will finish, if you provide 
uuid of existing one, cluster will remain.

- **Example 1. Use git repository and cluster definition:**
  
    ```python3
    from bodosdk.models import GitRepoSource, WorkspaceKeys, JobDefinition, JobClusterDefinition
    from bodosdk.client import get_bodo_client
    
    keys = WorkspaceKeys(
        client_id='XYZ',
        secret_key='XYZ'
    )
    client = get_bodo_client(keys)
    
    job_definition = JobDefinition(
        name='test',
        args='./examples/nyc-taxi/get_daily_pickups.py',
        source_config=GitRepoSource(
            repo_url='https://github.com/Bodo-inc/Bodo-examples.git',
            username='XYZ',
            token='XYZ'
        ),
        cluster_object=JobClusterDefinition(
            instance_type='c5.large',
            accelerated_networking=False,
            image_id='ami-0a2005b824a8758e5',
            workers_quantity=2
        ),
        variables={},
        timeout=120,
        retries=0,
        retries_delay=0,
        retry_on_timeout=False
    )
    
    client.job.create(job_definition)
    ```

- **Example 2. Run job from shared drive and existing cluster:**

  ```python3
  from bodosdk.models import JobCluster, WorkspaceSource, WorkspaceKeys, JobDefinition
  from bodosdk.client import get_bodo_client
  
  keys = WorkspaceKeys(
      client_id='XYZ',
      secret_key='XYZ'
  )
  client = get_bodo_client(keys)
  
  job_definition = JobDefinition(
      name='test',
      args='nyc-taxi/get_daily_pickups.py',
      source_config=WorkspaceSource(
          path='/shared/bodo-examples/examples/'
      ),
      cluster_object=JobCluster(
          uuid='0f0c5261-9827-4572-84f3-f6a9b10cf77d'
      ),
      variables={},
      timeout=120,
      retries=0,
      retries_delay=0,
      retry_on_timeout=False
  )
  
  client.job.create(job_definition)
  ```

  - **Example 3. Run job from a script file in an S3 bucket**
  
    To run a script file located on an S3 bucket, the cluster must have the required permissions to read the files from S3.
    This can be provided by creating an [Instance Role](https://pypi.org/project/bodosdk/#instance-role) with access to the required S3 bucket.
  
    Please make sure to specify an Instance Role that should be attached to the Job Cluster. The policy attached to the roles
    should provide access to both the bucket and its contents. 
    Also make sure to attach any other policies to this role for the cluster and the job to function correctly.
    This may include(but not limited to) s3 access for reading script files and s3 access to read data that is used in your job script file. 
  
    In addition to specifying the bucket path, we require users to specify the bucket region their bucket scripts are in in the S3Source
    definition, called bucket_region.
  
    ```python3
    from bodosdk.models import WorkspaceKeys, JobDefinition, JobClusterDefinition, S3Source, CreateRoleDefinition, CreateRoleResponse
    from bodosdk.client import get_bodo_client
    from typing import List
    from uuid import UUID
  
    keys = WorkspaceKeys(
        client_id='XYZ',
        secret_key='XYZ'
    )
    client = get_bodo_client(keys)
  
    role_definition = CreateRoleDefinition(
        name="test-sdk-role-creation",
        description="testing",
        data=InstanceRole(role_arn="arn:aws:iam::427443013497:role/testing_bucket_with_my_script")
    )
    result_create_role: CreateRoleResponse = client.instance_role.create(role_definition)
    # wait for successful role creation and then
    job_definition = JobDefinition(
        name='execute_s3_test',
        args='test_s3_job.py',
        source_config=S3Source(
            bucket_path='s3://path-to-my-bucket/my_job_script_folder/',
            bucket_region='us-east-1'
        ),
        cluster_object=JobClusterDefinition(
            instance_type='c5.large',
            accelerated_networking=False,
            image_id='ami-0a2005b824a8758e5',
            workers_quantity=2,
            instance_role_uuid=result_create_role.uuid
        ),
        variables={},
        timeout=120,
        retries=0,
        retries_delay=0,
        retry_on_timeout=False
    )
  
    client.job.create(job_definition)
    ```
  
    In the case you want to use one of the existing instance role that you might have pre-defined, you can copy the UUID for the instance
    role from the platform by navigating to the Instance role manager option in your workspace and add it to the SDK script or, 
    use the SDK to list all available instance roles, iterate through the list returned and break at the one we want to use depending on 
    a condition.
    ```python3
    from bodosdk.models import WorkspaceKeys, JobDefinition, JobClusterDefinition, S3Source, CreateRoleDefinition, CreateRoleResponse
    from bodosdk.client import get_bodo_client
    from typing import List
    from uuid import UUID
  
    keys = WorkspaceKeys(
        client_id='XYZ',
        secret_key='XYZ'
    )
    client = get_bodo_client(keys)
  
    list_of_instance_roles = client.instance_role.list()
    role_to_use = None
    for role in list_of_instance_roles:
        if role.name == 'role_i_want_to_use':
            role_to_use = role
            break
  
    # wait for successful role creation and then
    job_definition = JobDefinition(
        name='execute_s3_test',
        args='test_s3_job.py',
        source_config=S3Source(
            bucket_path='s3://path-to-my-bucket/my_job_script_folder/',
            bucket_region='us-east-1'
        ),
        cluster_object=JobClusterDefinition(
            instance_type='c5.large',
            accelerated_networking=False,
            image_id='ami-0a2005b824a8758e5',
            workers_quantity=2,
            instance_role_uuid=result_create_role.uuid
        ),
        variables={},
        timeout=120,
        retries=0,
        retries_delay=0,
        retry_on_timeout=False
    )
  
    client.job.create(job_definition)
    ```
  

<br/>
##### List jobs  
`BodoClient.job.list()`

Returns list of all jobs defined in workspace. 

**Example:**

```python
from typing import List
from bodosdk.models import WorkspaceKeys, JobResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
jobs: List[JobResponse] = client.job.list()
```


<br/>
##### Get job  

`BodoClient.job.get(job_uuid)`

Returns specific job in workspace. **Example:**

```python
from bodosdk.models import WorkspaceKeys, JobResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
job: JobResponse = client.job.get('8c32aec5-7181-45cc-9e17-8aff35fd269e')
```


<br/>
##### Remove job 

`BodoClient.job.delete(job_uuid)`

Removes specific job from workspace. **Example:**

```python
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
client.job.remove('8c32aec5-7181-45cc-9e17-8aff35fd269e')
```


<br/>
##### Get execution 

`BodoClient.job.get_job_executions(job_uuid)`

Gets all executions info for specific job. Result it's a list with one element (in future we might extend it)

```python
from bodosdk.models import WorkspaceKeys, JobExecution
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
executions: List[JobExecution] = client.job.get_job_executions('8c32aec5-7181-45cc-9e17-8aff35fd269e')
```

<br/>
##### Job waiter 

`BodoClient.job.get_waiter()`

Get waiter object, which can be used to wait till job finish. Waiter has following method

```python3
from typing import Callable
def wait(
        self,
        uuid,
        on_success: Callable = None,
        on_failure: Callable = None,
        on_timeout: Callable = None,
        check_period=10,
        timeout=None
):
  pass
```

By default, this returns job model if no callbacks are provided. There is option to pass callable objects as following
parameters:

- `on_success` - will be executed on success, job object passed as argument
- `on_failure` - will be executed on failure, job object passed as argument
- `on_timeout` - will be executed on timeout, job_uuid passed as argument

Other options are:

- `check_period` - seconds between status checks
- `timeout` - threshold in seconds after which Timeout error will be raised, `None` means no timeout

**Example 1. Success callback:**
```python
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
waiter = client.job.get_waiter()


def success_callback(job):
    print('Job has finished')
    return job


result = waiter.wait('8c32aec5-7181-45cc-9e17-8aff35fd269e', on_success=success_callback)
```

**Example 2. Timeout callback:**
```python
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
waiter = client.job.get_waiter()


def timeout_callback(job_uuid):
    print(f'Waiter timeout for {job_uuid}')
    return job_uuid


result = waiter.wait('8c32aec5-7181-45cc-9e17-8aff35fd269e', on_timeout=timeout_callback, timeout=1)
```

### Bodo Platform Batch jobs :material-alert-decagram:{ title="Recently Added" }

See [batch jobs][batch-jobs] for more information on Bodo Batch job concepts.


<br/>
##### Create batch job definition 

`BodoClient.job.create_batch_job_definition(job_definition: CreateBatchJobDefinition)`

Creates batch job definition in the given workspace. 

- **Example 1: Create batch job definition for a workspace source script**

    ```python
    from bodosdk.models import WorkspaceKeys, CreateBatchJobDefinition, BatchJobDefinition
    from bodosdk.client import get_bodo_client
    from bodosdk.models.job import CreateBatchJobDefinition, JobConfig, JobSource, JobSourceType, SourceCodeType, \
        WorkspaceDef, RetryStrategy
  
    keys = WorkspaceKeys(
      client_id='XYZ',
      secret_key='XYZ'
    )
    client = get_bodo_client(keys)
    
    workspace_source_def = JobSource(
        type=JobSourceType.WORKSPACE,
        definition=WorkspaceDef(
            path="/bodofs/Shared",
        ),
    )
    
    retry_strategy = RetryStrategy(
        num_retries=1,
        retry_on_timeout=False,
        delay_between_retries=2,
    )
    
    jobConfig = JobConfig(
        source=workspace_source_def,
        source_code_type=SourceCodeType.PYTHON,
        sourceLocation="test.py",
        args=None,
        retry_strategy=retry_strategy,
        timeout=10000,
        env_vars=None,
    )
    
    createBatchJobDef = CreateBatchJobDefinition(
        name="test-job",
        config=jobConfig,
        description="test-batch-job-description-attempt",
        cluster_config={
            "bodoVersion": "2023.1.9 (Preview)",
            "instance_type": "c5.2xlarge",
            "workers_quantity": 2,
            "accelerated_networking": False,
        }, )
    
    jobdef = client.job.create_batch_job_definition(createBatchJobDef)
    ```
  

- **Example 2: Create batch job definition for a git source script**

  ```python
  from bodosdk.models import WorkspaceKeys, CreateBatchJobDefinition, BatchJobDefinition
  from bodosdk.client import get_bodo_client
  from bodosdk.models.job import CreateBatchJobDefinition, JobConfig, JobSource, JobSourceType, SourceCodeType, \
      WorkspaceDef, RetryStrategy
    
  keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
  )
  client = get_bodo_client(keys)
    
  git_source_def = JobSource(
      type=JobSourceType.GIT,
      definition=GitDef(
          repo_url='https://github.com/Bodo-inc/Bodo-examples.git',
          username='XYZ',
          token='XYZ'
      ),
  )
    
  retry_strategy = RetryStrategy(
      num_retries=1,
      retry_on_timeout=False,
      delay_between_retries=2,
  )
    
  jobConfig = JobConfig(
      source=git_source_def,
      source_code_type=SourceCodeType.PYTHON,
      sourceLocation="test.py",
      args=None,
      retry_strategy=retry_strategy,
      timeout=10000,
      env_vars=None,
  )
    
  createBatchJobDef = CreateBatchJobDefinition(
      name="test-job",
      config=jobConfig,
      description="test-batch-job-description-attempt",
      cluster_config={
          "bodoVersion": "2023.1.9 (Preview)",
          "instance_type": "c5.2xlarge",
          "workers_quantity": 2,
          "accelerated_networking": False,
      }, )
    
  jobdef = client.job.create_batch_job_definition(createBatchJobDef)  
  ```
  

<br/>
##### List batch job definitions 

`BodoClient.job.list_batch_job_definitions()`

Lists all batch job definitions in the given workspace.

**Example:**
```python
from typing import List

from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobdefs: List[BatchJobDefinitionResponse] = client.job.list_batch_job_definitions()
```


<br/>
##### Get batch job definition by ID

`BodoClient.job.get_batch_job_definition(job_definition_id: str)`

Gets specific batch job definition by id.

**Example:**
```python
from typing import List

from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobdef: BatchJobDefinitionResponse = client.job.get_batch_job_definition('04412S5b-300e-42db-84d4-5f22f7506594')
```  

<br/>
##### Get batch job definition by name

`BodoClient.job.get_batch_job_definition_by_name(name: str)`

Gets specific batch job definition by id.

**Example:**
```python
from typing import List

from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobdef: BatchJobDefinitionResponse = client.job.get_batch_job_definition('04412S5b-300e-42db-84d4-5f22f7506594')
```  


<br/>
##### Remove batch job definition 

`BodoClient.job.remove_batch_job_definition(job_definition_id: str)`

Removes specific batch job definition by id.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
client.job.remove_batch_job_definition('04412S5b-300e-42db-84d4-5f22f7506594')
```


<br/>
##### Submit a batch job run 

`BodoClient.job.submit_batch_job_run(job_run: CreateJobRun)`

Submits a job run for a given batch job definition.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
client.job.submit_batch_job_run(CreateJobRun(batchJobDefinitionUUID='04412S5b-300e-42db-84d4-5f22f7506594', clusterUUID='12936Q5z-109d-89yi-23c4-3d91u1219823'))
```


<br/>
##### List batch job runs 

`BodoClient.job.list_batch_job_runs()`

+-------------------+---------------------+------------------------------------------+-------------------+
| Parameter         | Type                | Description                              | Required          |
+===================+=====================+==========================================+===================+
| `batch_job_id`    | `List[str]`         | List of Ids of the batch job definitions | No                |
+-------------------+---------------------+------------------------------------------+-------------------+
| `status`          | `List[JobRunStatus]`| List of Job Run Statuses                 | No                |
+-------------------+---------------------+------------------------------------------+-------------------+
| `cluster_id`      | `List[str]`         | List of Ids of the clusters              | No                |
+-------------------+---------------------+------------------------------------------+-------------------+

Lists all batch job runs in the given workspace filtered by given parameters.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobruns = client.job.list_batch_job_runs(status=[JobRunStatus.FAILED], cluster_id=['ba62e653-312a-490e-9457-71d7bc096959'])
```

<br/>
##### List batch job runs by batch job name

`BodoClient.job.list_batch_job_runs_by_names()`

+----------------------+---------------------+------------------------------------------+-------------------+
| Parameter            | Type                | Description                              | Required          |
+======================+=====================+==========================================+===================+
| `batch_job_names`    | `List[str]`         | List of Ids of the batch job definitions | No                |
+----------------------+---------------------+------------------------------------------+-------------------+
| `status`             | `List[JobRunStatus]`| List of Job Run Statuses                 | No                |
+----------------------+---------------------+------------------------------------------+-------------------+
| `cluster_id`         | `List[str]`         | List of Ids of the clusters              | No                |
+----------------------+---------------------+------------------------------------------+-------------------+

Lists all batch job runs in the given workspace filtered by given parameters.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client
from bodosdk.models.job import CreateJobRun, JobSource, JobRunStatus, BatchJobDefinitionResponse

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobruns = client.job.list_batch_job_runs_by_names(batch_job_names=['production-job-1'], status=[JobRunStatus.FAILED], cluster_id=['ba62e653-312a-490e-9457-71d7bc096959'])
```


<br/>
##### Get batch job run 


`BodoClient.job.get_batch_job_run(job_run_id: str)`

Gets specific batch job run by id.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
jobrun = client.job.get_batch_job_run('04412S5b-300e-42db-84d4-5f22f7506594')
```


<br/>
##### Cancel batch job run 


`BodoClient.job.cancel_batch_job_run(job_run_id: str)`


Cancels specific batch job run by id.


**Example:**

```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
client.job.cancel_batch_job_run('04412S5b-300e-42db-84d4-5f22f7506594')
```



<br/>
##### Check batch job run status 

`BodoClient.job.check_job_run_status(job_run_id: str)`

Checks status of specific batch job run by id.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)
status = client.job.check_job_run_status('04412S5b-300e-42db-84d4-5f22f7506594')
```


<br/>
##### Submit SQL job run {#sql-job-run}

`BodoClient.job.submit_sql_job_run(sql_job_details: CreateSQLJobRun)`

Submits a SQL query as a job run. 

!!! note
    This needs a database [catalog][catalog] to be configured in the workspace.

**Example:**
```python
from bodosdk.models import PersonalKeys, WorkspaceKeys, JobConfig, SourceCodeType, RetryStrategy, JobSourceType, \
    WorkspaceDef, CreateBatchJobDefinition
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id="XYZ",
    secret_key="XYZ"
)

client = get_bodo_client(keys)

jobrun = client.job.submit_sql_job_run(CreateSQLJobRun(
            clusterUUID=cluster.uuid,
            catalog="SNOWFLAKE_CATALOG",
            sqlQueryText="SELECT * FROM PUBLIC.TABLE LIMIT 10"))
```
<br/>

##### Job Run waiter {#job-waiter}

`BodoClient.job.get_job_run_waiter()`

Returns a waiter object that waits until the job run with the specified `uuid` finishes.
To wait for job run to be finished, invoke the `waiter.wait()` function,
which can take the following parameters.

```python3
from typing import Callable
def wait(
        self,
        uuid,
        on_success: Callable = None,
        on_failure: Callable = None,
        on_timeout: Callable = None,
        check_period=10,
        timeout=None
):
  pass
```

By default, it returns a job model if no callback is provided.  You can pass callable objects using the following
parameters:

- `on_success` - job object passed as argument to be executed on success
- `on_failure` - job object passed as argument to be executed on failure
- `on_timeout` - `job_uuid` passed as argument to be executed on timeout.

Other options are:

- `check_period` - seconds between status checks
- `timeout` - threshold in seconds after which Timeout error will be raised, `None` means no timeout

**Example 1. Success/Failure callbacks:**
```python
from bodosdk.models import WorkspaceKeys, CreateJobRun
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
input_job = CreateJobRun(clusterUUID='<cluster-uuid>', batchJobDefinitionUUID='<batch-job-definition-uuid>')
job_run = client.job.submit_batch_job_run(input_job)

waiter = client.job.get_job_run_waiter()

def success_callback(job):
    print("in success callback", job.status)

def failure_callback(job):
    print('in failure callback', job.status)

result = waiter.wait(job_run.uuid, on_success=success_callback, on_failure=failure_callback)
```

**Example 2. Timeout callback:**
```python
from bodosdk.models import WorkspaceKeys, CreateJobRun
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
input_job = CreateJobRun(clusterUUID='<cluster-uuid>', batchJobDefinitionUUID='<batch-job-definition-uuid>')
job_run = client.job.submit_batch_job_run(input_job)

waiter = client.job.get_job_run_waiter()

def timeout_callback(job_uuid):
    print(f'Waiter timeout for {job_uuid}')
    return job_uuid


result = waiter.wait(job_run.status, on_timeout=timeout_callback, timeout=1)
```


### Cluster resource {#cluster-resource}
Module responsible for managing clusters in workspace.

<!-- - [get available instance types](#available-instance-types) -->
<!-- - [get available images](#available-images) -->
<!-- - [create cluster](#create-cluster) -->
<!-- - [list clusters](#list-clusters) -->
<!-- - [get cluster](#get-cluster) -->
<!-- - [remove cluster](#remove-cluster) -->
<!-- - [scale cluster](#scale-cluster) -->
<!-- - [list jobs for cluster](#list-jobs-for-cluster) -->


<br/>
##### Available instance types {#available-instance-types}

`BodoClient.cluster.get_available_instance_types(region:str)`

Returns list of instance types available for given region

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
instance_types = client.cluster.get_available_instance_types('us-west-2')
```

<br/>
##### Available images {#available-images}

`BodoClient.cluster.get_available_images(region:str)`

Returns list of images available for given region

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
images = client.cluster.get_available_images('us-west-2')
```


<br/>
##### Create cluster {#create-cluster}

`BodoClient.cluster.create(cluster_definition: ClusterDefinition)`

Creates cluster in workspace.

```python3
from bodosdk.models import WorkspaceKeys, ClusterDefinition
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
cluster_definition = ClusterDefinition(
    name="test",
    instance_type="c5.large",
    workers_quantity=2,
    auto_shutdown=100,
    auto_pause=100,
    image_id="ami-038d89f8d9470c862",
    bodo_version="2022.4",
    description="my desc here"
)
result_create = client.cluster.create(cluster_definition)
```


<br/>
##### List clusters {#list-clusters}

```BodoClient.cluster.list()```

Returns list of all clusters in workspace

```python3
from bodosdk.models import WorkspaceKeys, ClusterResponse
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
clusters: List[ClusterResponse] = client.cluster.list()
```


<br/>
##### Get cluster {#get-cluster}

```BodoClient.cluster.get(cluster_uuid)```

Returns cluser by uuid

```python3
from bodosdk.models import WorkspaceKeys, ClusterResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
clusters: ClusterResponse = client.cluster.get('<CLUSTER-UUID>')
```


<br/>
##### Remove cluster {#remove-cluster}

```BodoClient.client.remove(cluster_uuid, force_remove=False, mark_as_terminated=False)```

Method removing cluster from platform
- force_remove: try to remove cluster even if something on cluster is happeing
- mark_as_terminated: mark cluster as removed without removing resources, may be useful if cluster creation failed and common removing is failing

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
client.cluster.remove('<CLUSTER-UUID>')
```


<br/>
##### Scale cluster {#scale-cluster}

```BodoClient.cluster.scale(scale_cluster: ScaleCluster)```

Changes number of nodes in cluster (AWS only)

```python3
from bodosdk.models import WorkspaceKeys, ScaleCluster, ClusterResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
NEW_WORKERS_QUANTITY = 3
scale_cluster = ScaleCluster(
    uuid='<CLUSTER-UUID>',
    workers_quantity=NEW_WORKERS_QUANTITY
)
cluster: ClusterResponse = client.cluster.scale(scale_cluster)
```


<br/>
##### List jobs for a cluster {#list-jobs-for-cluster}
```BodoClient.cluster.list_jobs(uuid)```

Gets all jobs for cluster

```python3
from bodosdk.models import WorkspaceKeys, JobResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
jobs: List[JobResponse] = client.cluster.list_jobs(uuid)
```

Get active jobs for cluster

```python3
from bodosdk.models import WorkspaceKeys, JobResponse, JobStatus
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
jobs: List[JobResponse] = client.cluster.list_jobs(uuid, status=[JobStatus.NEW, JobStatus.INPROGRESS])
```

<br/>
##### Modify Cluster metadata
`BodoClient.cluster.modify(uuid)`

This function can be used to edit cluster metadata for a given cluster. The properties that we can edit are description, autopause time, autoshutdown time and the number of workers. Changing the number of workers will kick off a scaling event on the cluster, which will resume the cluster if it is in paused state. The ModifyCluster object has the 4 properties as optional keys, meaning you can change just a subset of the 4 properties in a request too.

```python3
from bodosdk.models import WorkspaceKeys, ModifyCluster, ClusterResponse
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
modify_cluster = ModifyCluster(
    uuid=<cluster-uuid>,
    auto_pause=60,
    auto_shutdown=0,
    workers_quantity=4,
    description="using the SDK"
)
partial_modify_cluster = ModifyCluster(
    uuid=<cluster-uuid>,
    autopause=120,
)
new_cluster: List[ClusterResponse] = client.cluster.modify(modify_cluster)
new_cluster_partial: List[ClusterResponse] = client.cluster.modify(partial_modify_cluster)
```


### Workspace resource {#workspace-resource}
Module responsible for managing workspaces in an organization.

<!-- - [workspace getting started](#workspace-getting-started) -->
<!-- - [create workspace](#create-workspace) -->
<!-- - [list workspaces](#list-workspaces) -->
<!-- - [get workspace](#get-workspace) -->
<!-- - [remove workspace](#remove-workspace) -->
<!-- - [assign user](#assign-user) -->


<br/>
##### Workspace getting started {#workspace-getting-started}
In order to work with Workspace, users need to generate Personal Tokens, under Admin Console, from the Bodo Platform Dashboard.
Then instantiate a PersonalKeys object with the generated client_id and secret_id. Then Pass in this personal key while
instantiating a client object

```
from bodosdk.models import PersonalKeys
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
```


<br/>
##### Create Workspace {#create-workspace}
```BodoClient.workspace.create(workspace_definition: WorkspaceDefinition)```
Creates a workspace with the specifications passed in through a WorkspaceDefinition object under the 
user's organization
```
from bodosdk.models import PersonalKeys
from bodosdk.models import WorkspaceDefinition
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
wd = WorkspaceDefinition(
    name="<WORSPACE-NAME>",
    cloud_config_uuid="<CONFIG-UUID>",
    region="<WORKSPACE-REGION>"
)
resp = client.workspace.create(wd)
```


<br/>
##### List Workspaces {#list-workspaces}
```BodoClient.workspace.list()```
Returns a list of all workspaces defined under this organization. The with_task boolean controls printing out 
tasks running in the workspaces. The returned list is a list of GetWorkspaceResponse object
```
from bodosdk.models import PersonalKeys
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
resp = client.workspace.list(with_tasks=False)
```


<br/>
##### Get Workspace {#get-workspace}
```BodoClient.workspace.get(uuid: str)```
Returns information about the workspace with the given uuid. Returns a GetWorkspaceResponse object with details about the workspace uuid mentioned.
```
from bodosdk.models import PersonalKeys
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
resp = client.workspace.get("<WORKSPACE-UUID>")
```


<br/>
##### Remove Workspace {#remove-workspace}
```BodoClient.workspace.remove(uuid: str)```
Removes the workspace with the passed in uuid. The operation is only successful if all resources within the workspaces(jobs, clusters, notebooks) are terminated. Otherwise, returns an error. Returns None if successful
```
from bodosdk.models import PersonalKeys
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
resp = client.workspace.remove("<WORKSPACE-UUID>")
```


<br/>
##### Assign user {#assign-user}
```BodoClient.workspace.remove(uuid: str)```
Assign user to workspace.
```
from bodosdk.models import PersonalKeys
personal_keys = PersonalKeys(
    client_id='<CLIENT-ID>',
    secret_id='<SECRET-ID>',
)
client = get_bodo_organization_client(personal_keys)
workspace_uuid = "<some uuid>"
users: List[UserAssignment] = [
    UserAssignment(
        email="example@example.com",
        skip_email=True,
        bodo_role=BodoRole.ADMIN
    )
]
client.workspace.assign_users(workspace_uuid, users):
```


### Cloud Config {#cloud-config}
Module responsible for creating cloud configurations for organization.

<!-- - [create cloud configuration](#create-config) -->
<!-- - [list cloud configurations](#list-configs) -->
<!-- - [get cloud config](#get-config) -->


<br/>
##### Create config {#create-config}

```BodoClient.cloud_config.create(config: Union[CreateAwsCloudConfig, CreateAzureCloudConfig])```

Create cloud configuration for cloud

AWS example

```python3
from bodosdk.models import OrganizationKeys, CreateAwsProviderData, CreateAwsCloudConfig, AwsCloudConfig
from bodosdk.client import get_bodo_client

keys = OrganizationKeys(
    client_id='XYZ',
    secret_key='XYZ'
)

client = get_bodo_client(keys)

config = CreateAwsCloudConfig(
    name='test',
    aws_provider_data=CreateAwsProviderData(
        tf_backend_region='us-west-1',
        access_key_id='xyz',
        secret_access_key='xyz'
    )

)
config: AwsCloudConfig = client.cloud_config.create(config)
```

Azure example

```python3
from bodosdk.models import OrganizationKeys, CreateAzureProviderData, CreateAzureCloudConfig, AzureCloudConfig
from bodosdk.client import get_bodo_client

keys = OrganizationKeys(
    client_id='XYZ',
    secret_key='XYZ'
)

client = get_bodo_client(keys)

config = CreateAzureCloudConfig(
    name='test',
    azure_provider_data=CreateAzureProviderData(
        tf_backend_region='eastus',
        tenant_id='xyz',
        subscription_id='xyz',
        resource_group='MyResourceGroup'
    )
    
)
config: AzureCloudConfig = client.cloud_config.create(config)
```


<br/>
##### List configs {#list-configs}

```BodoClient.cloud_config.list()```

Get list of cloud configs.

```python3
from bodosdk.models import OrganizationKeys, AzureCloudConfig, AwsCloudConfig
from bodosdk.client import get_bodo_client
from typing import Union, List

keys = OrganizationKeys(
    client_id='XYZ',
    secret_key='XYZ'
)

client = get_bodo_client(keys)

configs: List[Union[AwsCloudConfig, AzureCloudConfig]] = client.cloud_config.list()
```




<br/>
##### Get config {#get-config}

```BodoClient.cloud_config.get(uuid: Union[str, UUID])```

Get cloud config by uuid.

```python3
from bodosdk.models import OrganizationKeys, AzureCloudConfig, AwsCloudConfig
from bodosdk.client import get_bodo_client
from typing import Union

keys = OrganizationKeys(
    client_id='XYZ',
    secret_key='XYZ'
)

client = get_bodo_client(keys)

config: Union[AwsCloudConfig, AzureCloudConfig] = client.cloud_config.get('8c32aec5-7181-45cc-9e17-8aff35fd269e')
```
### Instance Role Manager {#instance-role}
Module responsible for managing AWS roles in workspace.

<!-- - [create role](#create-role) -->
<!-- - [list roles](#list-roles) -->
<!-- - [get role](#get-role) -->
<!-- - [remove role](#remove-role) -->


<br/>
##### Create role {#create-role}

```BodoClient.instance_role.create()```

Creates an AWS role with the specified role definition with a given AWS role arn.

```python3
from bodosdk.models import WorkspaceKeys, CreateRoleDefinition, CreateRoleResponse
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
role_definition = CreateRoleDefinition(
    name="test-sdk-role-creation",
    description="testing",
    data=InstanceRole(role_arn="arn:aws:iam::1234567890:role/testing")
)
result_create:CreateRoleResponse = client.instance_role.create(role_definition)
```


<br/>
##### List roles {#list-roles}

```BodoClient.instance_role.list()```

Returns list of all roles in workspace

```python3
from bodosdk.models import WorkspaceKeys, InstanceRoleItem
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
result_list:List[InstanceRoleItem] = client.instance_role.list()
```


<br/>
##### Get role {#get-role}

```BodoClient.instance_role.get(cluster_uuid)```

Returns role by uuid

```python3
from bodosdk.models import WorkspaceKeys, InstanceRoleItem
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
clusters: InstanceRoleItem = client.instance_role.get('<CLUSTER-UUID>')
```


<br/>
##### Remove role {#remove-role}

```BodoClient.instance_role.remove(cluster_uuid, mark_as_terminated=False)```

Method removing role from a workspace
- mark_as_terminated: mark role as removed without removing resources, may be useful if role creation failed and common removing is failing

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from typing import List

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
client.instance_role.remove('<ROLE-UUID>')
```

## Catalog {#catalog}
Module responsible for storing database catalogs

<!-- - [create catalog](#create-catalog) -->
<!-- - [get catalog by uuid](#get-catalog-uuid) -->
<!-- - [get catalog by name](#get-catalog-name) -->
<!-- - [list catalogs](#list-catalogs) -->
<!-- - [update catalog](#update-catalog) -->
<!-- - [remove catalog](#remove-catalog-uuid) -->
<!-- - [remove all catalogs](#remove-all-catalogs) -->


### Create Catalog {#create-catalog}

```BodoClient.catalog.create()```

Stores the Database Catalog

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.catalog import CatalogDefinition, SnowflakeConnectionDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

# Type Support for Snowflake 
snowflake_definition = SnowflakeConnectionDefinition(
    host="test.snowflake.com",
    port=443,
    username="test-username",
    password="password",
    database="test-db",
    warehouse="test-wh",
    role="test-role"
)

# For other databases, need to defined as JSON
connection_data = {
    "host": "test.db.com",
    "username": "test-username",
    "password": "*****",
    "database": "test-db",
}

catalog_definition = CatalogDefinition(
    name="catalog-1",
    description="catalog description",
    catalogType="SNOWFLAKE", # Currently Support Snowflake 
    data=snowflake_definition
)

client.catalog.create(catalog_definition)


```

### Get Catalog by UUID {#get-catalog-uuid}

```BodoClient.catalog.get_catalog()```

Retrieves the Catalog details by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.catalog import CatalogInfo
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
catalog_info: CatalogInfo = client.catalog.get("<CATALOG-UUID>")
```

### Get Catalog by Name {#get-catalog-name}

```BodoClient.catalog.get_by_name()```

Retrieves the Catalog details by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.catalog import CatalogInfo
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
catalog_info: CatalogInfo = client.catalog.get_by_name("test-catalog")
```


<br/>
##### List Catalogs {#list-catalogs}

```BodoClient.catalog.list()```

Retrieves all catalogs in a workspace.

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.catalog import CatalogInfo
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
catalog_info: CatalogInfo = client.catalog.list()
```
### Update Catalog {#update-catalog}

```BodoClient.catalog.update()```

Updates the Database Catalog

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.catalog import CatalogDefinition, SnowflakeConnectionDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

# Type Support for Snowflake 
snowflake_definition = SnowflakeConnectionDefinition(
    host="update.snowflake.com",
    port=443,
    username="test-username",
    password="password",
    database="test-db",
    warehouse="test-wh",
    role="test-role"
)

new_catalog_def = CatalogDefinition(
    name="catalog-1",
    description="catalog description",
    catalogType="SNOWFLAKE", # Currently Support Snowflake 
    data=snowflake_definition
)
client.catalog.update("<CATALOG-UUID>", new_catalog_def)


```

### Remove Catalog by UUID {#remove-catalog-uuid}

```BodoClient.catalog.remove()```

Deletes a Database Catalog by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
client.catalog.remove("<CATALOG-UUID>")
```

### Remove all Catalogs {#remove-all-catalogs}

```BodoClient.catalog.remove()```

Deletes a Database Catalog by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
client.catalog.remove_all()
```

## Secret Groups {#secret-group}
Module responsible for separating secrets into multiple groups.

A default secret group will be created at the time of workspace creation.
Users can define custom secret groups using the following functions.

<!-- - [create secret group](#create-secret-group) -->
<!-- - [list secret groups](#list-secret-groups) -->
<!-- - [update secret group](#update-secret-group) -->
<!-- - [delete secret group](#delete-secret-group) -->

### Create Secret Group {#create-secret-group}

```BodoClient.secret_group.create()```

Create a secret group

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secret_group import SecretGroupDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

secret_group_definition = SecretGroupDefinition(
    name="sg-1", # Name should be unique to that workspace
    description="secret group description",
)

client.secret_group.create(secret_group_definition)
```

### List Secret Groups {#list-secret-groups}

```BodoClient.secret_group.list()```

List all the secret groups in a workspace.

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secret_group import SecretGroupInfo
from typing import List
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
groups_list: List[SecretGroupInfo] = client.secret_group.list()
```

### Update Secret Group {#update-secret-group}

```BodoClient.secret_group.update()```

Updates the secret group description

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secret_group import SecretGroupInfo, SecretGroupDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

update_secret_group_def = SecretGroupDefinition(
    name="sg-1", # Cannot modify the name in the group
    description="secret group description",
)
groups_data: SecretGroupInfo = client.secret_group.update(update_secret_group_def)
```
### Delete Secret Group {#delete-secret-group}

```BodoClient.secret_group.remove()```

Removes the secret group.

**Note: Can only remove if all the secrets in the group are deleted**

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

client.secret_group.remove("<secret-group-uuid>")
```

## Secrets {#secrets}
Module responsible for creating secrets.


<!-- - [create secret](#create-secrets) -->
<!-- - [get secret](#get-secret) -->
<!-- - [list secrets](#list-secrets) -->
<!-- - [list secrets by secret group](#list-secrets-by-secret-group) -->
<!-- - [update secret](#update-secret) -->
<!-- - [delete secret](#delete-secret) -->


### Create Secret {#create-secret}

```BodoClient.secrets.create()```

Create the secret in a secret group.

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

secret_definition = SecretDefinition(
    name="secret-1",
    data={
        "key": "value"
    },
    secret_group="<secret-group-name>" #If not defined, defaults to default to secret group
)

client.secrets.create(secret_definition)
```

### Get Secrets by UUID {#get-secret}

```BodoClient.secrets.get()```

Retrieves the Secrets by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretInfo
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
secret_info: SecretInfo = client.secrets.get("<secret-uuid>")
```

### List Secrets by Workspace {#list-secrets}

```BodoClient.secrets.list()```

List the secrets in a workspace 

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretInfo
from typing import List
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
secrets_info: List[SecretInfo] = client.secrets.list()
```

### List Secrets by Secret Group {#list-secrets-by-secret-group}

```BodoClient.secrets.list_by_group()```

List the Secrets by Secret Group

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretInfo
from typing import List
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
secrets_info: List[SecretInfo] = client.secrets.list_by_group("<secret-group-name>")
```


### Update Secret {#update-secret}

```BodoClient.secrets.update()```

Updates the secret.

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretDefinition
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)

update_secret_def = SecretDefinition(
    data={
        "key": "value"
    }
)

client.secrets.update("<secret-uuid>", update_secret_def)
```

### Delete Secrets by UUID {#delete-secret}

```BodoClient.secrets.remove()```

Delete the Secret by UUID

```python3
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client
from bodosdk.models.secrets import SecretInfo
keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
secret_info: SecretInfo = client.secrets.remove("<secret-uuid>")
```
