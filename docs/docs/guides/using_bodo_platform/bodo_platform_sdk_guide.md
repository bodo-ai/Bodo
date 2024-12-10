# Bodo Platform SDK Guide

Bodo Platform SDK is a Python library that provides a simple way to interact with the Bodo Platform API. It allows you
to create, manage, and monitor resources such as clusters, jobs, and workspaces.

## Getting Started

### Installation

```shell
pip install bodosdk
```

### Create a workspace client

First you need to access your workspace in `https://platform.bodo.ai/` and create an _API Token_ in the Bodo Platform.
This token is used to authenticate your client with the Bodo Platform API.

Navigate to _API Tokens_ in the Admin Console to generate a token.
Copy and save the token's _Client ID_ and _Secret Key_ and use them to define a client (`BodoClient`) that can interact
with the Bodo Platform.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)
```

Alternatively, set `BODO_CLIENT_ID` and `BODO_SECRET_KEY` environment variables to avoid storing keys in your code.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
```

You can access the `workspace_data` attribute of the client to get more information about your workspace:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
print(my_workspace.workspace_data)
```

### Additional Configuration Options for `BodoClient`

- `print_logs`: defaults to False. All API requests and responses are printed to the console if set to True.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(print_logs=True)
```

### Create a cluster

This example creates a simple one node cluster in your workspace with the latest available bodo version.
It returns a cluster object.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='My first cluster',
    instance_type='c5.large',
    workers_quantity=1
)
```

### Wait for status

You can use the `wait_for_status` method to wait until the cluster ready to use.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='My first cluster',
    instance_type='c5.large',
    workers_quantity=1
)
my_cluster.wait_for_status(['RUNNING'])
```

This method will wait until either the cluster reaches any of the provided status, or fails (goes into the `FAILED`
status) for any reason.

### Update your cluster

Now let's update our cluster. While the cluster is `RUNNING`, you can update
the `name`, `description`, `auto_pause`, `auto_stop`,
and `workers_quantity`(this will trigger scaling) fields only:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='My first cluster',
    instance_type='c5.large',
    workers_quantity=1
)
my_cluster.wait_for_status(['RUNNING'])
my_cluster.update(
    description='My description',
    name="My updated cluster",
    auto_pause=15,
    auto_stop=30
)
```

All other modifications like `instance_type`, `bodo_version` etc. need the cluster to be `STOPPED` first.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.get("cluster_id")
if my_cluster.status != 'STOPPED':
    my_cluster.stop(wait=True)
my_cluster.update(instance_type='c5.2xlarge', workers_quantity=2)
```

### Create a Job

Lets now create a job on our `RUNNING` cluster.
First, access `https://platform.bodo.ai` and navigate to the jupyter notebook in your workspace. Then
create the following `test.py` file in your main directory:

```python
import pandas as pd
import numpy as np
import bodo
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000

df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")
time.sleep(1)  # wait till file will be available on all nodes


@bodo.jit(cache=True)
def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df1 = df[df.B > 4].A.sum()
    print("Execution time:", time.time() - t1)
    return df1


result = computation()
print(result)
```

Now you can define a job on cluster through SDK, wait till it has `SUCCEEDED` and check its logs as follows:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.get("cluster_id")
my_job = my_cluster.run_job(
    code_type='PYTHON',
    source={'type': 'WORKSPACE', 'path': '/'},
    exec_file='test.py'
)
print(my_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

You can use almost the same configuration to run a SQL file. All you need is to define your `test.sql` file and a
Catalog on `https://platform.bodo.ai`:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.get("cluster_id")
my_job = my_cluster.run_job(
    code_type='SQL',
    source={'type': 'WORKSPACE', 'path': '/'},
    exec_file='test.sql',
    catalog="MyCatalog"
)
print(my_job.wait_for_status(['SUCCEEDED']).get_stdout())

```

### Cluster List and executing jobs on multiple clusters

Now let's try to run same job on different clusters:

```python
from bodosdk import BodoWorkspaceClient
import random

my_workspace = BodoWorkspaceClient()

random_val = random.random()  # just to avoid conflicts on name
clusters_conf = [('c5.large', 8), ('c5.xlarge', 4), ('c5.2xlarge', 2)]
for i, conf in enumerate(clusters_conf):
    my_workspace.ClusterClient.create(
        name=f'Test {i}',
        instance_type=conf[0],
        workers_quantity=conf[1],
        custom_tags={'test_tag': f'perf_test{random_val}'}  # let's add tag to easy filter our clusters
    )
# get list by tag
clusters = my_workspace.ClusterClient.list(filters={
    'tags': {'test_tag': f'perf_test{random_val}'}
})
# run same job 3 times, once per each cluster
jobs = clusters.run_job(
    code_type='PYTHON',
    source={'type': 'WORKSPACE', 'path': '/'},
    exec_file='test.py'
)
# wait for jobs to finish and print results
for job in jobs.wait_for_status(['SUCCEEDED']):
    print(job.name, job.cluster.name)
    print(job.get_stdout())
# remove our clusters
jobs.clusters.delete()  # or clusters.delete()
```

### Execute SQL query

You can also execute SQL queries by passing just query text like following:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_sql_job = my_workspace.JobClient.run_sql_query(sql_query="SELECT 1", catalog="MyCatalog", cluster={
    "name": 'Temporary cluster',
    "instance_type": 'c5.large',
    "workers_quantity": 1
})
print(my_sql_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

In this case, when you provide a cluster configuration rather than an existing cluster, the created cluster will be
terminated as soon
as SQL job finishes.

If you want to execute a SQL job on existing cluster, you can use `run_sql_query` on cluster:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='My cluster',
    instance_type='c5.large',
    workers_quantity=1
)
my_sql_job = my_cluster.run_sql_query(sql_query="SELECT 1", catalog="MyCatalog")
print(my_sql_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

### Connector

You can also execute SQL queries using a cluster Connector via a Cursor:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='My cluster',
    instance_type='c5.large',
    workers_quantity=1
)
connection = my_cluster.connect(
    'MyCatalog')  # or connection = my_workspace.ClusterClient.connect('MyCatalog', 'cluster_id')
print(connection.cursor().execute("SELECT 1").fetchone())
my_cluster.delete()
```

### Job Templates

Rather than defining jobs from scratch every time, you can create a template for your jobs.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
tpl = my_workspace.JobTemplateClient.create(
    name='My template',
    cluster={
        'instance_type': 'c5.xlarge',
        'workers_quantity': 1
    },
    code_type="SQL",
    catalog="MyCatalog",
    exec_text="SELECT 1"
)
job1 = tpl.run()  # you can simply run it
job2 = tpl.run(exec_text="SELECT 2")  # or run it with overriding template values
job3 = tpl.run(cluster={'instance_type': 'c5.large'})  # you can override even part of cluster configuration

jobs = my_workspace.JobClient.list(filters={'template_ids': [tpl.id]})  # you can filter jobs by its template_id
for job in jobs.wait_for_status(['SUCCEEDED']):
    print(job.name, job.cluster.instance_type, job.get_stdout())

```

You can also run your template on specific clusters:

```python
from bodosdk import BodoWorkspaceClient
from bodosdk.models import JobTemplateFilter

my_workspace = BodoWorkspaceClient()
tpls = my_workspace.JobTemplateClient.list(filters=JobTemplateFilter(names=['My template']))
my_cluster = my_workspace.ClusterClient.create(
    name='My cluster',
    instance_type='c5.large',
    workers_quantity=1
)
print(my_cluster.run_job(template_id=tpls[0].id).wait_for_status(['SUCCEEDED']).get_stdout())
my_cluster.delete()
```

## Statuses

Each resource, Cluster, Job or Workspace has own set of statuses which are as follows:

### Cluster

- NEW
- INPROGRESS
- PAUSING
- PAUSED
- STOPPING
- STOPPED
- INITIALIZING
- RUNNING
- FAILED
- TERMINATED

### Job

- PENDING
- RUNNING
- SUCCEEDED
- FAILED
- CANCELLED
- CANCELLING
- TIMEOUT

### Workspace

- NEW
- INPROGRESS
- READY
- FAILED
- TERMINATING
- TERMINATED

## Organization client and workspaces

To manage workspaces, you need different keys (generated for the organization) and different SDK clients.
Let's list all our workspaces:

```python
from bodosdk import BodoOrganizationClient

my_org = BodoOrganizationClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)  # or BodoOrganizationClient() if `BODO_ORG_CLIENT_ID` and `BODO_ORG_SECRET_KEY` are exported
for w in my_org.list_workspaces():
    print(w.name)
```

You can filter workspaces with valid filters:

```python
from bodosdk import BodoOrganizationClient
from bodosdk.models import WorkspaceFilter

my_org = BodoOrganizationClient()

for w in my_org.list_workspaces(filters=WorkspaceFilter(statuses=['READY'])):
    print(w.name)
```

You can provide filters such as `WorkspaceFilter` imported from `bodosdk.models` or as a dictionary:

```python
from bodosdk import BodoOrganizationClient

my_org = BodoOrganizationClient()

for w in my_org.list_workspaces(filters={"statuses": ['READY']}):
    print(w.name)
```

### Create new Workspace

```python
from bodosdk import BodoOrganizationClient

my_org = BodoOrganizationClient()
my_workspace = my_org.create_workspace(
    name="SDK test",
    region='us-east-2',
    cloud_config_id="a0d1242c-3091-42de-94d9-548e2ae33b73",
    storage_endpoint_enabled=True
).wait_for_status(['READY'])
assert my_workspace.id == my_org.list_workspaces(filters={"names": ['SDK test'], "statuses": ['READY']})[0].id
my_workspace.delete()  # remove workspace at the end
```

### Upgrade workspace infra

In some cases when you have a workspace created a while ago, you may want to re-run terraform to
apply fresh changes to workspace infrastructure. You can do it as follows:

```python
from bodosdk import BodoOrganizationClient

my_org = BodoOrganizationClient()
my_org.list_workspaces(filters={'ids': ['workspace_to_update1_id', 'workspace_to_update2_id']}).update_infra()
```

## Advanced

In this section we will present more examples of bodosdk usage.

### Workspace created in existing VPC

Its possible to create workspace on existing infrastructure. The only requirement is that the VPC needs access to the
Internet, either NAT or IGW. Bodo platform needs this to allow clusters to be authorized via an external authorization
service.

```python
from bodosdk import BodoOrganizationClient

my_org = BodoOrganizationClient()
my_workspace = my_org.create_workspace(
    cloud_config_id="cloudConfigId",
    name="My workspace",
    region="us-east-1",
    storage_endpoint_enabled=True,
    vpc_id="existing-vpc-id",
    private_subnets_ids=['subnet1', 'subnet2'],
    public_subnets_ids=['subnet3']
)
my_workspace.wait_for_status(['READY'])
```

### Spot instances and Auto-AZ

You can create a cluster using spot instances, to reduce cost of usage. A limitation of using spot instances
is that you cannot pause this type of cluster. Further cluster may be occasionally become unavailable (when spot
instance is released).

Auto-AZ is a mechanism which retries cluster creation in another Availability zone, when current availability zone does
not
have enough instances of desired type.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.create(
    name='Spot cluster',
    instance_type='c5.large',
    workers_quantity=1,
    use_spot_instance=True,
    auto_az=True,
)
```

### Accelerated networking

Accelerated networking is enabled by default for instances that support it.

You can get a list of all supported instances using the `ClusterClient.get_instances` function. This returns a list of
InstanceType objects. The field `accelerated_networking` tells you whether network acceleration is enabled.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()

accelerated_networking_instances = [x for x in my_workspace.ClusterClient.get_instances() if x.accelerated_networking]

my_cluster = my_workspace.ClusterClient.create(
    name='Spot cluster',
    instance_type=accelerated_networking_instances[0].name,
    workers_quantity=1,
)
```

### Preparing clusters for future use

A cluster may be suspended in two states: `PAUSED` and `STOPPED`.
Spot clusters cannot be `PAUSED`. There are 3 differences between those states: cost, start up time, error rate.

#### Costs

`PAUSED` > `STOPPED` - The`PAUSED` state incurs disk costs while the `STOPPED` state doesn't.

#### Start up time

`STOPPED` > `PAUSED` - Resuming machines in the `PAUSED` state is much faster than restarting `STOPPED` clusters, as
those machines are already
defined in the cloud provider.

#### Error rate

`PAUSED` > `STOPPED` - The error rate is the likelihood of the situation where number of available instances
of desired types is lower than number of requested workers.

In `PAUSED` state, instance entities are already defined.
Upon resuming a `PAUSED` cluster, since we request for the cluster resources all at once, it's more likely to run into
capacity issues than in a `STOPPED` state.
In a `STOPPED` state, instance creation is managed by ASG, which will try to create instances in a staggered manner.

In the following example, we prepare a cluster for future use by pausing it once its created:

```python
from bodosdk import BodoWorkspaceClient
from bodosdk.models import ClusterFilter

my_workspace = BodoWorkspaceClient()

clusters_conf = {
    'Team A': {
        'instance_type': 'c5.2xlarge',
        'workers': 4,
    },
    'Team b': {
        'instance_type': 'c5.xlarge',
        'workers': 2,
    },
    'Team C': {
        'instance_type': 'c5.16xlarge',
        'workers': 2,
    }
}
for owner, conf in clusters_conf.items():
    my_workspace.ClusterClient.create(
        name=f"{owner} Cluster",
        instance_type=conf['instance_type'],
        workers_quantity=conf['workers'],
        custom_tags={'owner': owner, 'purpose': 'test'}
    )

my_workspace.ClusterClient.list(
    filters=ClusterFilter(tags={'purpose': 'test'})
).wait_for_status(
    ['RUNNING', 'INITIALIZING']
).pause().wait_for_status(['PAUSED'])
```

### Run a job using a cluster as template

Let's imagine that you have a cluster (in any state) and you want to run job on the same specification, but you don't
want to use the previously defined cluster. You can use the existing cluster as a template for the cluster to run the
job:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient()
my_cluster = my_workspace.ClusterClient.get('existing_cluster')
cluster_conf = my_cluster.dict()
del cluster_conf['uuid']
my_sql_job = my_workspace.JobClient.run_sql_query(sql_query="SELECT 1", catalog="MyCatalog", cluster=cluster_conf)
```

A new cluster will be created with the same configuration as the existing cluster, and the job will be run on it. The
new cluster will be terminated as soon as the job finishes.

!!! seealso "See Also"
    [BodoSDK Reference](../../api_docs/platform_sdk.md)