# bodo.rebalance

`bodo.rebalance(data, dests=None, random=False, random_seed=None, parallel=False)`
Manually redistribute data evenly across [selected] ranks.

### Arguments

- ``data``: data to rebalance.
- ``dests``: selected ranks to distribute data to. By default, distribution includes all ranks.
- ``random``: flag to randomize order of the rows of the data. Default: `False`.
- ``random_seed``: number to initialize random number generator.
- ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

### Example Usage

- Example with just the `parallel` flag set to `True`:

    ```py
    import bodo
    import pandas as pd

    @bodo.jit
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df = df.sort_values("power")[df["power"] > 400]
        print(df.shape)
        df = bodo.rebalance(df, parallel=True)
        print("After rebalance: ", df.shape)

    mean_power()
    ```

    Save code in ``test_rebalance.py`` file and run with 4 processes.

    ```shell
    BODO_NUM_WORKERS=4 python test_rebalance.py
    ```

    ```console
    [stdout:0]
    (5, 10)
    After rebalance: (33, 10)
    [stdout:1]
    (18, 10)
    After rebalance: (33, 10)
    [stdout:2]
    (82, 10)
    After rebalance: (33, 10)
    [stdout:3]
    (26, 10)
    After rebalance: (32, 10)
    ```

- Example to distribute the data from all ranks to subset of ranks using ``dests`` argument.

    !!! note
        The following example uses [SPMD launch mode][spmd].


    ```py

    import bodo
    import pandas as pd

    @bodo.jit(spawn=False)
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df = df.sort_values("power")[df["power"] > 400]
        return df

    df = mean_power()
    print(df.shape)
    df = bodo.rebalance(df, dests=[1,3], parallel=True)
    print("After rebalance: ", df.shape)
    ```
    Save code in ``test_rebalance.py`` file and run with 4 processes.

    ```shell
    mpiexec -n 4 python test_rebalance.py
    ```

    Output:

    ```console
    [stdout:0]
    (5, 10)
    After rebalance: (0, 10)
    [stdout:1]
    (18, 10)
    After rebalance: (66, 10)
    [stdout:2]
    (82, 10)
    After rebalance: (0, 10)
    [stdout:3]
    (26, 10)
    After rebalance: (65, 10)
    ```

