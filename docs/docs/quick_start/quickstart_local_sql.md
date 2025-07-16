<!--
NOTE: the examples in this file are covered by tests in bodosql/tests/test_quickstart_docs.py. Any changes to examples in this file should also update the corresponding unit test(s).
 -->

# Bodo SQL Quick Start {#quickstart-local-sql}

This quickstart guide will walk you through the process of running a simple SQL query using Bodo on your local machine.

## Prerequisites

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install -U bodo` or `conda install bodo -c conda-forge`).
Additionally, install bodosql with pip or conda:

```shell
pip install bodosql
```

```shell
conda install bodosql -c bodo.ai -c conda-forge
```

## Generate Sample Data

Let's start by creating a Parquet file with some sample data. The following Python code creates a Parquet file with two columns `A` and `B` and 20 million rows. The column `A` contains values from 0 to 29, and the column `B` contains values from 0 to 19,999,999.

```python
import pandas as pd
import numpy as np
import bodo
import bodosql

NUM_GROUPS = 30
NUM_ROWS = 20_000_000
df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")
```

## Create a local in-memory SQL Table

Now let's create a local in-memory SQL table from the Parquet file. We can use the [`TablePATH` API][tablepath-api] to register the table into our [`BodoSQLContext`][bodosqlcontext-api].

```python
bc = bodosql.BodoSQLContext(
    {
        "TABLE1": bodosql.TablePath("my_data.pq", "parquet")
    }
)
```

## Write a SQL Query

Now we can write a SQL query to compute the sum of column `A` for all rows where `B` is greater than 4.

```python
df1 = bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")
print(df1)
```


## Running your code

Bringing it all together, the complete code looks like this:

```python
import pandas as pd
import numpy as np
import bodosql

NUM_GROUPS = 30
NUM_ROWS = 20_000_000

df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})

df.to_parquet("my_data.pq")

bc = bodosql.BodoSQLContext(
    {
        "TABLE1": bodosql.TablePath("my_data.pq", "parquet")
    }
)

df1 = bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")
print(df1)
```


To run the code, save it to a file, e.g. `test_bodo_sql.py`, and run the following command in your terminal:

```bash
python test_bodo_sql.py
```


By default Bodo will use all available cores. To set a limit on the number of processes spawned, set the environment variable `BODO_NUM_WORKERS`.
Check the [SQL API Reference][bodosql] for the full list of supported SQL operations.
