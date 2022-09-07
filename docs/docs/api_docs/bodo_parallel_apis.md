Bodo Parallel APIs {#bodoparallelapis}
==================

This page lists advanced parallel APIs provided by Bodo for  
finer-grained control over data distribution and processes.


## bodo.allgatherv
    
<code><apihead>bodo.<apiname>allgatherv</apiname>(data, warn_if_rep=True)</apihead></code><br><br><br>Gather data from all ranks and send to all, effectively replicating the data.

***Arguments***

- ``data``: data to gather.
- ``warn_if_rep``: prints a BodoWarning if data to gather is replicated. 

***Example Usage***
    
```py

import bodo
import pandas as pd

@bodo.jit
def mean_power():
    df = pd.read_parquet("data/cycling_dataset.pq")
    return bodo.allgatherv(df)

df = mean_power()
print(df)
```

Save code in ``test_allgatherv.py`` file and run with `mpiexec`.

```shell
mpiexec -n 4 python test_allgatherv.py
```

Output:

```console
[stdout:0]
      Unnamed: 0    altitude  cadence  ...  power  speed                time
0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
...          ...         ...      ...  ...    ...    ...                 ...
3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

[3902 rows x 10 columns]
[stdout:1]
      Unnamed: 0    altitude  cadence  ...  power  speed                time
0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
...          ...         ...      ...  ...    ...    ...                 ...
3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

[3902 rows x 10 columns]
[stdout:2]
      Unnamed: 0    altitude  cadence  ...  power  speed                time
0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
...          ...         ...      ...  ...    ...    ...                 ...
3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

[3902 rows x 10 columns]
[stdout:3]
      Unnamed: 0    altitude  cadence  ...  power  speed                time
0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
...          ...         ...      ...  ...    ...    ...                 ...
3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

[3902 rows x 10 columns]
```

## bodo.barrier

<code><apihead>bodo.<apiname>barrier</apiname>()</apihead></code><br><br><br><br> 
Synchronize all processes. Block process from proceeding until all processes reach this point.

***Example Usage***

A typical example is to make sure all processes see side effects simultaneously.
For example, a process can delete files from storage while
others wait before writing to file:

```py 
import shutil, os
import numpy as np

# remove file if exists
if bodo.get_rank() == 0:
    if os.path.exists("data/data.pq"):
        shutil.rmtree("data/data.pq")

# make sure all processes are synchronized
# (e.g. all processes need to see effect of rank 0's work)
bodo.barrier()


@bodo.jit
def f(n):
    df = pd.DataFrame({"A": np.arange(n)})
    df.to_parquet("data/data.pq")


f(10)
```

The following figure illustrates what happens when processes call
`bodo.barrier()`. When barrier is called, a process pauses and waits
until all other processes have reached the barrier:

![Process synchronization with Barrier](../img/barrier.svg#center)

!!! danger
    The example above shows that it is possible to have each process
    follow a different control flow, but all processes must always call
    the same Bodo functions in the same order.

## bodo.gatherv

<code><apihead>bodo.<apiname>gatherv</apiname>(data, allgather=False, warn_if_rep=True, root=0)</apihead></code><br><br><br><br> 
Collect distributed data manually by *gathering* them into a single rank. 

***Arguments***
  
- ``data``: data to gather.
- ``root``: specify rank to collect the data. Default: rank `0`.
- ``warn_if_rep``: prints a BodoWarning if data to gather is replicated. 
- ``allgather``: send gathered data to all ranks. Default: `False`. Same behavior as ``bodo.allgatherv``.

***Example Usage***
    
```py
import bodo
import pandas as pd

@bodo.jit
def mean_power():
    df = pd.read_parquet("data/cycling_dataset.pq")
    return bodo.gatherv(df, root=1)

df = mean_power()
print(df)
```
Save code in ``test_gatherv.py`` file and run with `mpiexec`.

```shell
mpiexec -n 4 python test_gatherv.py
```

Output:

```console
[stdout:1]
      Unnamed: 0    altitude  cadence  ...  power  speed                time
0              0  185.800003       51  ...     45  3.459 2016-10-20 22:01:26
1              1  185.800003       68  ...      0  3.710 2016-10-20 22:01:27
2              2  186.399994       38  ...     42  3.874 2016-10-20 22:01:28
3              3  186.800003       38  ...      5  4.135 2016-10-20 22:01:29
4              4  186.600006       38  ...      1  4.250 2016-10-20 22:01:30
...          ...         ...      ...  ...    ...    ...                 ...
3897        1127  178.199997        0  ...      0  3.497 2016-10-20 23:14:31
3898        1128  178.199997        0  ...      0  3.289 2016-10-20 23:14:32
3899        1129  178.199997        0  ...      0  2.969 2016-10-20 23:14:33
3900        1130  178.399994        0  ...      0  2.969 2016-10-20 23:14:34
3901        1131  178.399994        0  ...      0  2.853 2016-10-20 23:14:35

[3902 rows x 10 columns]
[stdout:0]
Empty DataFrame
Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
Index: []

[0 rows x 10 columns]
[stdout:2]
Empty DataFrame
Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
Index: []

[0 rows x 10 columns]
[stdout:3]
Empty DataFrame
Columns: [Unnamed: 0, altitude, cadence, distance, hr, latitude, longitude, power, speed, time]
Index: []

[0 rows x 10 columns]
```

## bodo.get_rank 

<code><apihead>bodo.<apiname>get_rank</apiname>()</apihead></code><br><br><br>
Get the process number from Bodo (called `rank` in MPI terminology).

***Example Usage***

    
Save following code in `get_rank.py` file and run with `mpiexec`.

```py
import bodo
# some work only on rank 0
if bodo.get_rank() == 0:
    print("rank 0 done")

# some work on every process
print("rank", bodo.get_rank(), "here")
```

```console 
mpiexec -n 4 python get_rank.py
```

Output

```console
rank 0 done
rank 0 here
rank 1 here
rank 2 here
rank 3 here
```

## bodo.get_size 


<code><apihead>bodo.<apiname>get_size</apiname>()</apihead> </code><br><br><br>Get the total number of processes.

***Example Usage***
    
Save following code in `get_rank_size.py` file and run with `mpiexec`.

```py
import bodo
# some work only on rank 0
if bodo.get_rank() == 0:
    print("rank 0 done")

# some work on every process
print("rank", bodo.get_rank(), "here")
print("total ranks:", bodo.get_size())
```

```console 
mpiexec -n 4 python get_rank_size.py
```

Output

```console
rank 0 done
rank 0 here
total ranks: 4
rank 1 here
total ranks: 4
rank 2 here
total ranks: 4
rank 3 here
total ranks: 4
```

## bodo.random_shuffle
    
<code><apihead>bodo.<apiname>random_shuffle</apiname>(data, seed=None, dests=None, parallel=False)</apihead></code><br><br><br>Manually shuffle data evenly across selected ranks.

***Arguments***
    
- ``data``: data to shuffle.
- ``seed``: number to initialze random number generator.
- ``dests``: selected ranks to distribute shuffled data to. By default, distribution includes all ranks.
- ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

***Example Usage***
    
```py
import bodo
import pandas as pd

@bodo.jit
def test_random_shuffle():
    df = pd.DataFrame({"A": range(100)})
    return df

df = test_random_shuffle()
print(df.head())
df = bodo.random_shuffle(res, parallel=True)
print(df.head())
```

Save code in ``test_random_shuffle.py`` file and run with `mpiexec`.

```shell
mpiexec -n 4 python test_random_shuffle.py
```

Output:

```console
[stdout:1]
    A
0  25
1  26
2  27
3  28
4  29
    A
19  19
10  10
17  42
9    9
17  17
[stdout:3]
    A
0  75
1  76
2  77
3  78
4  79
    A
6   31
0   25
24  49
22  22
5   30
[stdout:2]
    A
0  50
1  51
2  52
3  53
4  54
    A
11  36
24  24
15  65
14  14
10  35
[stdout:0]
    A
0  0
1  1
2  2
3  3
4  4
    A
4   29
18  18
8   58
15  15
3   28
```

## bodo.rebalance

<code><apihead>bodo.<apiname>rebalance</apiname>(data, dests=None, random=False, random_seed=None, parallel=False)</apihead> </code><br><br><br>Manually redistribute data evenly across [selected] ranks.

***Arguments***
    
- ``data``: data to rebalance.
- ``dests``: selected ranks to distribute data to. By default, distribution includes all ranks.
- ``random``: flag to randomize order of the rows of the data. Default: `False`.
- ``random_seed``: number to initialize random number generator.
- ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

***Example Usage*** 

- Example with just the `parallel` flag set to `True`:
 
    ```py
    import bodo
    import pandas as pd
    
    @bodo.jit
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df = df.sort_values("power")[df["power"] > 400]
        return df
    
    df = mean_power()
    print(df.shape)
    df = bodo.rebalance(df, parallel=True)
    print("After rebalance: ", df.shape)
    ```

    Save code in ``test_rebalance.py`` file and run with `mpiexec`.
    
    ```shell
    mpiexec -n 4 python test_rebalance.py
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

    ```py
    
    import bodo
    import pandas as pd
    
    @bodo.jit
    def mean_power():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df = df.sort_values("power")[df["power"] > 400]
        return df
    
    df = mean_power()
    print(df.shape)
    df = bodo.rebalance(df, dests=[1,3], parallel=True)
    print("After rebalance: ", df.shape)
    ```
    Save code in ``test_rebalance.py`` file and run with `mpiexec`.
    
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

## bodo.scatterv

<code><apihead>bodo.<apiname>scatterv</apiname>(data, warn_if_dist=True)</apihead></code><br><br><br><br> 
Distribute data manually by *scattering* data from one process to all processes.

***Arguments***

- ``data``: data to distribute.
- ``warn_if_dist``: flag to print a BodoWarning if ``data`` is already distributed.

!!! note 
    Currently, `bodo.scatterv` only supports scattering from rank 0.

***Example Usage***

- When used outside of JIT code, we recommend that the argument be set to ``None`` for all ranks except rank 0. 
  For example:
  
  ```py 
  import bodo
  import pandas as pd
  
  
  @bodo.jit(distributed=["df"])
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
    [repo](https://github.com/Bodo-inc/Bodo-tutorial).

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
  
  @bodo.jit()
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

