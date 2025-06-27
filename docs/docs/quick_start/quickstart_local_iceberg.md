<!-- 
NOTE: the examples in this file are covered by tests in bodo/tests/test_quickstart_docs.py. Any changes to examples in this file should also update the corresponding unit test(s).
 -->

# Bodo Iceberg Quick Start {#quickstart-local-iceberg}

This quickstart guide will walk you through the process of creating and reading from an Iceberg table using Bodo on your local machine.

## Installation

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install -U bodo` or `conda install bodo -c bodo.ai -c conda-forge`).


## Create a Local Iceberg Table with Bodo DataFrame Library

This example demonstrates simple write of a table on the filesystem without a catalog:

```python
import bodo.pandas as pd
import numpy as np

n = 20_000_000
df = pd.DataFrame({"A": np.arange(n) % 30, "B": np.arange(n)})
df.to_iceberg("test_table", location="./iceberg_warehouse")
```

Now let's read the Iceberg table:

```python
print(pd.read_iceberg("test_table", location="./iceberg_warehouse"))
```

See [DataFrame Library API reference][dataframe-lib] for more information.
Note that this quickstart uses a local Iceberg table, but you can also use Bodo with Iceberg tables on S3, ADLS, and GCS as well.


## Amazon S3 Tables

[Amazon S3 Tables](https://aws.amazon.com/s3/features/tables/) simplify Iceberg use
and table maintenance by providing builtin Apache Iceberg support.
Bodo supports S3 Tables in both Python and SQL seamlessly.
Here is a step by step example for using S3 Tables in Bodo.

Make sure you have your environment ready:

1. Create a Table bucket on S3 (not a regular bucket).
   You can simply use the console with [this link](https://us-east-2.console.aws.amazon.com/s3/table-buckets?region=us-east-2) (replace region if desired).
2. Make sure you have AWS credentials in your environment (e.g. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`).
3. Make sure the user associated with your credentials has `AmazonS3TablesFullAccess` policy attached. You can use IAM in the AWS console (e.g. [this link](https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-2#/home)).
4. Set default region to the bucket region in the environment. For example:
    ```bash
    export AWS_REGION="us-east-2"
    ```
5. Make sure you have the latest AWS CLI (see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
   since this is a new feature and create a namespace in the table bucket. For example (replace region, account number and bucket name):
    ```bash
    aws s3tables create-namespace --table-bucket-arn arn:aws:s3tables:us-east-2:111122223333:bucket/my-test-bucket --namespace my_namespace
    ```

Now you are ready to use Bodo to read and write S3 Tables. Run this example code (replace bucket name, account ID, region, namespace):

```python
import pandas as pd
import numpy as np
import bodo

BUCKET_NAME="my-test-bucket"
ACCOUNT_ID="111122223333"
REGION="us-east-2"
NAMESPACE="my_namespace"
CONN_STR=f"iceberg+arn:aws:s3tables:{REGION}:{ACCOUNT_ID}:bucket/{BUCKET_NAME}"

NUM_GROUPS = 30
NUM_ROWS = 20_000_000


@bodo.jit
def example_write_iceberg_table():
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_sql(
        name="my_table_1",
        con=CONN_STR,
        schema=NAMESPACE,
        if_exists="replace"
    )

example_write_iceberg_table()

@bodo.jit
def example_read_iceberg():
    df = pd.read_sql_table(
            table_name="my_table_1",
            con=CONN_STR,
            schema=NAMESPACE
         )
    print(df)
    return df

df_read = example_read_iceberg()
print(df_read)
```

You can use BodoSQL to work with S3 Tables as well. Here is a simple example:

```python
import pandas as pd
import bodosql

BUCKET_NAME="my-test-bucket"
ACCOUNT_ID="111122223333"
REGION="us-east-2"
NAMESPACE="my_namespace"
ARN_STR=f"arn:aws:s3tables:{REGION}:{ACCOUNT_ID}:bucket/{BUCKET_NAME}"

catalog = bodosql.S3TablesCatalog(ARN_STR)
bc = bodosql.BodoSQLContext(catalog=catalog)
df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
bc = bc.add_or_replace_view("TABLE1", df)

query = f"""
CREATE OR REPLACE TABLE "{NAMESPACE}"."my_table" AS SELECT * FROM __bodolocal__.table1
"""
bc.sql(query)

df_read = bc.sql(f"SELECT * FROM \"{NAMESPACE}\".\"my_table\"")
print(df_read)
```
