# bodo.random_shuffle

`bodo.random_shuffle(data, seed=None, dests=None, parallel=False)`
Manually shuffle data evenly across selected ranks.

### Arguments

- ``data``: data to shuffle.
- ``seed``: number to initialze random number generator.
- ``dests``: selected ranks to distribute shuffled data to. By default, distribution includes all ranks.
- ``parallel``: flag to indicate whether data is distributed. Default: `False`. Inside JIT default value depends on Bodo's distribution analysis algorithm for the data passed (For more information, see Data Distribution section below).

### Example Usage

Note that this example uses [SPMD launch mode][spmd].

```py
import bodo
import pandas as pd

@bodo.jit(spawn=False)
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

