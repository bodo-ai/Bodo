<!-- 
NOTE: the examples in this file are covered by tests in bodo/tests/test_quickstart_docs.py. Any changes to examples in this file should also update the corresponding unit test(s).
 -->

# Bodo Python Quick Start {#quickstart-local-python}

This quickstart guide will walk you through the process of running a simple Python computation using Bodo on your local machine.

## Prerequisites

[Install Bodo](../installation_and_setup/install.md) to get started (e.g., `pip install bodo` or `conda install bodo -c bodo.ai -c conda-forge`).

## Generate Sample Data

Let's start by creating a Parquet file with some sample data. The following Python code creates a Parquet file with two columns `A` and `B` and 20 million rows. The column `A` contains values from 0 to 29, and the column `B` contains values from 0 to 19,999,999.

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
```

## A Simple Pandas Computation

Now let's write a simple Python function that computes the division of column `B` over column `A` (when `A` is not zero) using pandas. We decorate the function with `@bodo.jit` to indicate that we want to compile the code using Bodo. Let's also add a timer to measure the execution time.


```python
@bodo.jit(cache=True)
def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df2 = pd.DataFrame({"A": df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)})
    df2.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation()
```

## Running the Code

Bringing it all together, the complete code looks like this:

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

@bodo.jit(cache=True)
def computation():
    t1 = time.time()
    df = pd.read_parquet("my_data.pq")
    df2 = pd.DataFrame({"A": df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)})
    df2.to_parquet("out.pq")
    print("Execution time:", time.time() - t1)

computation()
```

To run the code, save it to a file, e.g. `test_bodo.py`, and run the following command in your terminal:

```bash
python test_bodo.py
```

By default Bodo will use all available cores. To set a limit on the number of processes spawned, set the environment variable `BODO_NUM_WORKERS`.
Note that the first time you run this code, it may take a few seconds to compile the code.
Next time you run the code, it will execute much faster. Check the [Python API Reference][pythonreference] for the full list of supported Python operations.


## Bodo Integration Quickstart

Here are high level steps for integrating Bodo into Python workloads:

1. Installation and Import

    a. [Install Bodo](../installation_and_setup/install.md) (e.g. `pip install bodo`).

    b. In each file where parallelization is desired, add:
        ```python
        import bodo
        ```

2. JIT-Compile Your Main Processing Functions

    a. Decorate computationally intensive functions with `@bodo.jit(cache=True)`.

    b. Gather all configuration inputs (e.g., file lists) outside of the jitted function and pass those inputs into the jitted function as parameters.

    * Any file discovery logic (e.g., `glob.glob`, or referencing environment variables) should happen _outside_ your jitted function.

    * The jitted function should receive the final list of files/tables and other necessary parameters explicitly.

    c. Keep I/O of large data inside JIT functions if the storage format is supported by Bodo (Parquet, CSV, JSONL, Iceberg, Snowflake).

    Example code:

    ```python
    @bodo.jit(cache=True)
    def process_data(file_list):
       df = pd.read_parquet(file_list)
       print(df.A.sum())

    file_list = get_file_list()
    process_data(file_list)
    ```

3. Avoid Python Features Incompatible With Bodo

    Inside jitted functions, avoid:

    - Using list/set/dict data structures for large data.
    - Unusual dynamic Python features (e.g., closures capturing changing state).
    - Unused imports or library calls that Bodo cannot compile.

    Generally, use Pandas DataFrames and Numpy arrays for large data and use idiomatic Pandas code to process data.


4. Use `@bodo.wrap_python` for Calling Non-JIT Libraries inside JIT

    A common pattern is calling a Python function using a domain-specific library on every row of a dataframe. Use `@bodo.wrap_python` for this case.
    Provide output type of the Python call using a sample of representative output:

    ```python
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    out_list_type = bodo.typeof([1, 2])

    @bodo.wrap_python(out_list_type)
    def run_tokenizer(text):
        tokenized = tokenizer(text)
        return tokenized["input_ids"]

    @bodo.jit
    def preprocess_pile(file_list):
        df = pd.read_parquet(file_list)
        df["input_ids"] = df["text"].map(run_tokenizer)
        ...
    ```

See [Compilation Tips and Troubleshooting][compilation] for more tips on handling compilation issues.
