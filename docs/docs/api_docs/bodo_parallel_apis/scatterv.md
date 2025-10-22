# bodo.scatterv

`bodo.scatterv(data, warn_if_dist=True)`
<br>
Distribute data manually by *scattering* data from one process to all processes.

### Arguments

- ``data``: data to distribute.
- ``warn_if_dist``: flag to print a BodoWarning if ``data`` is already distributed.

!!! note
      Currently, `bodo.scatterv` only supports scattering from rank 0.

!!! note
      The following examples use [SPMD launch mode][spmd].

### Example Usage

- When used outside of JIT code, we recommend that the argument be set to ``None`` for all ranks except rank 0.
  For example:

  ```py
  import bodo
  import pandas as pd


  @bodo.jit(spawn=False, distributed=["df"])
  def mean_power(df):
      x = df.power.mean()
      return x

  df = None
  # only rank 0 reads the data
  if bodo.get_rank() == 0:
      df = pd.read_parquet("data/cycling_dataset.pq")

  df = bodo.scatterv(df)
  res = mean_power(df)
  print(res)
  ```

  Save the code in ``test_scatterv.py`` file and run with `mpiexec`.

  ```shell
  mpiexec -n 4 python test_scatterv.py
  ```

  Output:

  ```console
  [stdout:0] 102.07842132239877
  [stdout:1] 102.07842132239877
  [stdout:2] 102.07842132239877
  [stdout:3] 102.07842132239877
  ```

!!! note
    `data/cycling_dataset.pq` is located in the Bodo tutorial
    [repo](https://github.com/bodo-ai/Bodo-tutorial).

- This is not a strict requirement. However, since this might be bad practice in certain situations,
  Bodo will throw a warning if the data is not None on other ranks.

  ```py
  import bodo
  import pandas as pd

  df = pd.read_parquet("data/cycling_dataset.pq")
  df = bodo.scatterv(df)
  res = mean_power(df)
  print(res)
  ```

  Save code in ``test_scatterv.py`` file and run with `mpiexec`.

  ```shell
  mpiexec -n 4 python test_scatterv.py
  ```

  Output:

  ```console
  BodoWarning: bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. This data won't be sent to any other ranks and will be overwritten with data from rank 0.

  [stdout:0] 102.07842132239877
  [stdout:1] 102.07842132239877
  [stdout:2] 102.07842132239877
  [stdout:3] 102.07842132239877
  ```

- When using ``scatterv`` inside of JIT code, the argument must have the same type on each rank due to Bodo's typing constraints.
  All inputs except for rank 0 are ignored.

  ```py
  import bodo
  import pandas as pd

  @bodo.jit(spawn=False)
  def impl():
      if bodo.get_rank() == 0:
          df = pd.DataFrame({"A": [1,2,3,4,5,6,7,8]})
      else:
          df = pd.DataFrame({"A": [-1]*8})
      return bodo.scatterv(df)
  print(impl())
  ```

  Save code in ``test_scatterv.py`` file and run with `mpiexec`.

  ```shell
  mpiexec -n 8 python test_scatterv.py
  ```

  Output:

  ```console
  [stdout:6]
        A
  6     7
  [stdout:0]
        A
  0     1
  [stdout:1]
        A
  1     2
  [stdout:4]
        A
  4     5
  [stdout:7]
        A
  7     8
  [stdout:3]
        A
  3     4
  [stdout:2]
        A
  2     3
  [stdout:5]
        A
  5     6
  ```


!!! note
    `scatterv`, `gatherv`, `allgatherv`, `rebalance`, and `random_shuffle` work with all distributable data types. This includes:

    -   All supported numpy array types.
    -   All supported pandas array types (with the exception of Interval Arrays).
    -   All supported pandas Series types.
    -   All supported DataFrame types.
    -   All supported Index types (with the exception of Interval Index).
    -   Tuples of the above types.
