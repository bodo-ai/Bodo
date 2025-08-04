# bodo.pandas.BodoSeries.map_partitions
``` py
BodoSeries.map_partitions(func, *args, **kwargs) -> BodoSeries | BodoDataFrame
```

Apply a function to groups of rows in a Series and return a Series or Series of the same size.

If the input Series is lazy (i.e. its plan has not been evaluated yet) and *func* returns a Series, then
the output will be lazy as well. When the lazy output is evaluated, *func* will take batches of
rows from the input Series. In the cases where *func* returns a Series or the input Series is not lazy,
each worker will call *func* on their entire local chunk of the input Series.

<p class="api-header">Parameters</p>

: __func : *Callable*:__ A function that takes in a Series and returns a Series or DataFrame (with the same number of rows). Currently, functions that return a DataFrame will trigger execution even if the input Series has a lazy plan.

: __\*args:__ Additional positional arguments to pass to *func*.

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.

<p class="api-header">Returns</p>

: __BodoSeries__ or __BodoDataFrame__:  The result of applying *func* to the BodoSeries.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bs = bd.Series(
    range(15)
   )

bs_mapped = bs.map_partitions(lambda ser: ser + 15)
print(bs_mapped)
```

Output:
```
0     15
1     17
2     19
3     21
4     23
5     25
6     27
7     29
8     31
9     33
10    35
11    37
12    39
13    41
14    43
dtype: int64[pyarrow]
```

---
