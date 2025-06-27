# bodo.pandas.BodoSeries.map
```
BodoSeries.map(arg, na_action=None) -> BodoSeries
```
Map values of a BodoSeries according to a mapping.

!!! note
    Calling `BodoSeries.map` will immediately execute a plan to generate a small sample of the BodoSeries
    and then call `pandas.Series.map` on the sample to infer output types
    before proceeding with lazy evaluation.

<p class="api-header">Parameters</p>

: __arg : *function, collections.abc.Mapping subclass or Series*:__ Mapping correspondence.

: __na_actions : *{None, ‘ignore’}, default None*:__ If 'ignore' then NaN values will be propagated without passing them to the mapping correspondence.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": bd.array([4, 5, 6, -1] * 3, "Int64"),
    }
)

bodo_ser = bdf.A.map(lambda x: x ** 2)
print(type(bodo_ser))
print(bodo_ser)
```

Output:
```
<class 'bodo.pandas.series.BodoSeries'>
0      1
1      4
2      9
3     49
4      1
5      4
6      9
7     49
8      1
9      4
10     9
11    49
Name: A, dtype: int64[pyarrow]
```

---
