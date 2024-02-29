# bodo.gatherv

`bodo.gatherv(data, allgather=False, warn_if_rep=True, root=0)`
<br> 
Collect distributed data manually by *gathering* them into a single rank. 

### Arguments
  
- ``data``: data to gather.
- ``root``: specify rank to collect the data. Default: rank `0`.
- ``warn_if_rep``: prints a BodoWarning if data to gather is replicated. 
- ``allgather``: send gathered data to all ranks. Default: `False`. Same behavior as ``bodo.allgatherv``.

### Example Usage
    
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

