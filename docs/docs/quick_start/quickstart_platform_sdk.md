# Bodo Platform SDK Quick Start {#quickstart-platform-sdk}

This quickstart guide will walk you through running a job on the Bodo Platform using the Bodo Platform SDK installed on your local machine.

## Getting Started

### Installation

```shell
pip install bodosdk
```

### Create a workspace client

To authenticate with the Bodo Platform API, you need to create an API Token:

1. Log in to your workspace at https://platform.bodo.ai/.
2. Navigate to API Tokens in the Admin Console. 
3. Generate a token and copy the Client ID and Secret Key.

Use these credentials to define a `BodoWorkspaceClient` for interacting with the platform:

```python
from bodosdk import BodoWorkspaceClient

my_workspace = BodoWorkspaceClient(
    client_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    secret_key="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
)
```

### Create a cluster
Create a single-node cluster in your workspace with the latest available Bodo version:

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

### Run Python Job

**Step 1: Write the Job Script**

Access https://platform.bodo.ai and open the Jupyter notebook in your workspace. Create the following test.py file in your main directory:

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

**Step 2: Run the Job**

Use the SDK to run the job on your cluster, wait for it to SUCCEED, and check its logs:


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


### Run a SQL Job
To run a SQL job, create a test.sql file and a catalog in https://platform.bodo.ai. Then, run the job as follows:
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

my_cluster = my_workspace.ClusterClient.get("cluster_id")

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

my_cluster = my_workspace.ClusterClient.get("cluster_id")

# Connect and execute query
connection = my_cluster.connect('MyCatalog')
result = connection.cursor().execute("SELECT 1").fetchone()
print(result)
```


!!! seealso "See Also"
 * [BodoSDK Guide](../../guides/using_bodo_platform/bodo_platform_sdk_guide)
 * [BodoSDK Reference](../../api_docs/platform_sdk)
