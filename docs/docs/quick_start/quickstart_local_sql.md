# Bodo SQL Quickstart (Local) {#quickstart-local-sql}

This quickstart guide will walk you through the process of running a simple SQL query using Bodo on your local machine using a local SQL table.


## Prerequisites

[Conda](https://docs.conda.io){target="blank"} is the recommended way to install Bodo on your local environment. You can install the _Community Edition_ using conda, which allows you to use Bodo for free on up to 8 cores. 


```console 
conda create -n Bodo python=3.12 -c conda-forge
conda activate Bodo
conda install bodosql -c bodo.ai -c conda-forge
```

This command creates a conda environment called `Bodo` to run your code in and installs Bodo Community Edition by default, which is free and works on up to 8 cores.


## Generate Sample Data

Let's start by creating a parquet file with some sample data. The following Python code creates a parquet file with two columns `A` and `B` and 20 million rows. The column `A` contains values from 0 to 29, and the column `B` contains values from 0 to 19,999,999.

```python
import pandas as pd
import numpy as np
import bodo
import bodosql
import time

NUM_GROUPS = 30
NUM_ROWS = 20_000_000
df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")
```

## Create a local SQL Table

Now let's create a local SQL table from the parquet file. We can use the [`TablePATH` API][tablepath-api] to register the table into our [`BodoSQLContext`][bodosqlcontext-api].

```python
bc = bodosql.BodoSQLContext(
    {
        "TABLE1": bodosql.TablePath("my_data.pq", "parquet")
    }
)
```

## Write a SQL Query

Now we can write a SQL query to compute the sum of column `A` for all rows where `B` is greater than 4. Since we are using a BodoSQLContext locally, we need to encapsulate the statement within a `@bodo.jit` decorated function to indicate that we want to compile the code using Bodo. Let's also add a timer to measure the execution time.

```python
@bodo.jit(cache=True)
def query(bc):
    t1 = time.time()
    df1 = bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")
    # print the execution time only on one core
    print("Execution time:", time.time() - t1)
    return df1

result = query(bc)
print(result)
```


## Running your code

Bringing it all together, the complete code looks like this:

```python
import pandas as pd
import numpy as np
import bodo
import bodosql
import time

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

@bodo.jit(cache=True)
def query(bc):
    t1 = time.time()
    df1 = bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")
    print("Execution time:", time.time() - t1)
    return df1

result = query(bc)
print(result)
```


To run the code, save it to a file, e.g. `test_bodo_sql.py`, and run the following command in your terminal:

```bash
mpiexec -n 8 python test_bodo_sql.py
```

Replace `8` with the number of cores you want to use. Note that the first time you run this code, it may take a few seconds to compile the code. Next time you run the code, it will execute much faster. Check the [SQL API Reference][bodosql] for the full list of supported SQL operations.





