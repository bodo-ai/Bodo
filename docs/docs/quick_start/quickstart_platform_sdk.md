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
    instance_type='c6i.large',
    workers_quantity=1
)
my_cluster.wait_for_status(['RUNNING'])
print(my_cluster.id)
```

### Run Python Job

**Step 1: Write the Job Script**

Access https://platform.bodo.ai and open the Jupyter notebook in your workspace. Create the following test.py file in your main directory:

```python
import bodo
import time
import numpy as np

@bodo.jit
def calc_pi(n):
    t1 = time.time()
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    pi = 4 * np.sum(x**2 + y**2 < 1) / n
    print("Execution time:", time.time()-t1, "\nresult:", pi)

calc_pi(2 * 10**6)
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
To run a SQL job, create a test.sql file and a [catalog][sql_catalog] in https://platform.bodo.ai. Then, run the job as follows:
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
 * [BodoSDK PyPi](https://pypi.org/project/bodosdk/)