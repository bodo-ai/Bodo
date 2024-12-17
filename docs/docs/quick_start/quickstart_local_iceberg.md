# Bodo Iceberg Quick Start {#quickstart-local-iceberg}

This quickstart guide will walk you through the process of creating and reading from an Iceberg table using Bodo on your local machine.

## Prerequisites

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install bodo` or `conda install bodo -c bodo.ai -c conda-forge`).
Additionally, install bodo-iceberg-connector with pip or conda:

```shell
pip install bodo-iceberg-connector
```

```shell
conda install -c bodo.ai bodo-iceberg-connector
```


## Create a local Iceberg Table


Now let's create a function to create an Iceberg table from a sample DataFrame with two columns (`A` and `B`) and 20 million rows.
Column `A` contains values from 0 to 29, and column `B` contains values from 0 to 19,999,999.

Our function will write data to a local directory called _MY_DATABASE_. The table named _MY_TABLE_ will be stored under the _MY_SCHEMA_ schema (which is a folder under _MY_DATABASE_).


```python
import pandas as pd
import numpy as np
import bodo


NUM_GROUPS = 30
NUM_ROWS = 20_000_000


@bodo.jit
def example_write_iceberg_table():
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_sql(
        name="MY_TABLE",
        con="iceberg://MY_DATABASE",
        schema="MY_SCHEMA",
        if_exists="replace"
    )

example_write_iceberg_table()
```

## Read the Iceberg Table

We can read the Iceberg table to make sure it was created correctly. 

```python
@bodo.jit
def example_read_iceberg():
    df = pd.read_sql_table(
            table_name="MY_TABLE",
            con="iceberg://MY_DATABASE",
            schema="MY_SCHEMA"
         )
    print(df)
    return df


df_read = example_read_iceberg()
```



## Running your code

Bringing it all together, the complete code looks like this:

```python
import pandas as pd
import numpy as np
import bodo


NUM_GROUPS = 30
NUM_ROWS = 20_000_000


@bodo.jit
def example_write_iceberg_table():
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_sql(
        name="MY_TABLE",
        con="iceberg://MY_DATABASE",
        schema="MY_SCHEMA",
        if_exists="replace"
    )

example_write_iceberg_table()

@bodo.jit
def example_read_iceberg():
    df = pd.read_sql_table(
            table_name="MY_TABLE",
            con="iceberg://MY_DATABASE",
            schema="MY_SCHEMA"
         )
    print(df)
    return df

df_read = example_read_iceberg()
```


To run the code, save it to a file, e.g. `test_bodo_iceberg.py`, and run the following command in your terminal:

```bash
python test_bodo_iceberg.py
```


By default Bodo will use all available cores. To set a limit on the number of processes spawned, set the environment variable `BODO_NUM_WORKERS`.
Within the JIT functions data will be distributed across the number of cores you specify. Once data is returned, it can be accessed as if it all exists on a single process, though under the hood Bodo will only transfer the full data to the main process if it is actually used.
E.g. if you run the code with 8 cores, here's the expected print out:

<details> <summary> Click to expand output</summary>

    ```console
              A         B
    15000000  0  15000000
    15000001  1  15000001
    15000002  2  15000002
    15000003  3  15000003
    15000004  4  15000004
    ...      ..       ...
    17499995  5  17499995
    17499996  6  17499996
    17499997  7  17499997
    17499998  8  17499998
    17499999  9  17499999
    
    [2500000 rows x 2 columns]         
    
               A         B
    17500000  10  17500000
    17500001  11  17500001
    17500002  12  17500002
    17500003  13  17500003
    17500004  14  17500004
    ...       ..       ...
    19999995  15  19999995
    19999996  16  19999996
    19999997  17  19999997
    19999998  18  19999998
    19999999  19  19999999
    
    [2500000 rows x 2 columns]         
    
             A        B
    7500000  0  7500000
    7500001  1  7500001
    7500002  2  7500002
    7500003  3  7500003
    7500004  4  7500004
    ...     ..      ...
    9999995  5  9999995
    9999996  6  9999996
    9999997  7  9999997
    9999998  8  9999998
    9999999  9  9999999
    
    [2500000 rows x 2 columns]
    
               A         B
    12500000  20  12500000
    12500001  21  12500001
    12500002  22  12500002
    12500003  23  12500003
    12500004  24  12500004
    ...       ..       ...
    14999995  25  14999995
    14999996  26  14999996
    14999997  27  14999997
    14999998  28  14999998
    14999999  29  14999999
    
    [2500000 rows x 2 columns]
    
              A        B
    2500000  10  2500000
    2500001  11  2500001
    2500002  12  2500002
    2500003  13  2500003
    2500004  14  2500004
    ...      ..      ...
    4999995  15  4999995
    4999996  16  4999996
    4999997  17  4999997
    4999998  18  4999998
    4999999  19  4999999
    
    [2500000 rows x 2 columns]
    
               A         B
    10000000  10  10000000
    10000001  11  10000001
    10000002  12  10000002
    10000003  13  10000003
    10000004  14  10000004
    ...       ..       ...
    12499995  15  12499995
    12499996  16  12499996
    12499997  17  12499997
    12499998  18  12499998
    12499999  19  12499999
    
    [2500000 rows x 2 columns]          
    
              A        B
    5000000  20  5000000
    5000001  21  5000001
    5000002  22  5000002
    5000003  23  5000003
    5000004  24  5000004
    ...      ..      ...
    7499995  25  7499995
    7499996  26  7499996
    7499997  27  7499997
    7499998  28  7499998
    7499999  29  7499999
    
    [2500000 rows x 2 columns]
    
             A        B
    0        0        0
    1        1        1
    2        2        2
    3        3        3
    4        4        4
    ...     ..      ...
    2499995  5  2499995
    2499996  6  2499996
    2499997  7  2499997
    2499998  8  2499998
    2499999  9  2499999
    
    [2500000 rows x 2 columns]
    ```
</details>

Note that this quickstart uses a local Iceberg table, but you can also use Bodo with Iceberg tables on S3, ADLS, and GCS as well.

