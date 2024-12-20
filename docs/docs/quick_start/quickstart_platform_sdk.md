# Bodo Platform SDK Quick Start {#quickstart-platform-sdk}

This quickstart guide will walk you through the process of running a Job on Bodo platform using Bodo Platform SDK,
installed on your local machine.

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

### Create a cluster
Creates one node cluster in your workspace with the latest available bodo version.

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)

my_cluster = my_workspace.ClusterClient.create(
    name='My first cluster',
    instance_type='c5.large',
    workers_quantity=1
)
```

### Create a Job

Lets now create a job on our `RUNNING` cluster.
First, access `https://platform.bodo.ai` and navigate to the Jupyter notebook in your workspace. Then
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
    df2 = pd.DataFrame({"A": df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)})
    df2.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation()
```

Now you can define a job on cluster through SDK, wait till it has `SUCCEEDED` and check its logs as follows:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)

my_cluster = my_workspace.ClusterClient.get("cluster_id")
my_job = my_cluster.run_job(
    code_type='PYTHON',
    source={'type': 'WORKSPACE', 'path': '/'},
    exec_file='test.py'
)

# Print stdout from job
print(my_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

You can use almost the same configuration to run a SQL file. All you need is to define your `test.sql` file and a
Catalog on `https://platform.bodo.ai`:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)

my_cluster = my_workspace.ClusterClient.get("cluster_id")
my_job = my_cluster.run_job(
    code_type='SQL',
    source={'type': 'WORKSPACE', 'path': '/'},
    exec_file='test.sql',
    catalog="MyCatalog"
)

# Print stdout from job
print(my_sql_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

### Execute SQL query
Execute SQL queries by passing just query text like following:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)

my_cluster = my_workspace.ClusterClient.create(
    name='My cluster',
    instance_type='c5.large',
    workers_quantity=1
)

# Execute query
my_sql_job = my_cluster.run_sql_query(sql_query="SELECT 1", catalog="MyCatalog")

# Print stdout from job
print(my_sql_job.wait_for_status(['SUCCEEDED']).get_stdout())
```

### Connector
Execute SQL queries using a cluster Connector via a Cursor:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)

# Create simple cluster
my_cluster = my_workspace.ClusterClient.create(
    name='My cluster',
    instance_type='c5.large',
    workers_quantity=1
)

# Connect and execute query
connection = my_cluster.connect('MyCatalog')
result = connection.cursor().execute("SELECT 1").fetchone()
print(result)

# Cleanup cluster if needed
my_cluster.delete()
```


!!! seealso "See Also"
 * [BodoSDK Guide](../../guides/bodo_platform_sdk_guide.md)
 * [BodoSDK Reference](../../api_docs/platform_sdk.md)
