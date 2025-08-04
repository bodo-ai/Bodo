# bodo.pandas.BodoSeries.map
```
BodoSeries.map(arg, na_action=None, engine="bodo") -> BodoSeries
```
Map values of a BodoSeries according to a mapping.

If `arg` is a function, bodo.jit will be applied to `arg`.  If this JIT compilation fails for any
reason, the mapping function will be run as a normal Python function.  If the compilation succeeds,
the JIT compiled function will be used for the map and the overheads associated with running Python code
from within the execution pipeline are avoided.

!!! note
    Calling `BodoSeries.map` will immediately execute a plan if this JIT compilation fails to
    generate a small sample of the BodoSeries and then call `pandas.Series.map` on the sample to
    infer output types before proceeding with lazy evaluation.

!!! note
    Functions passed to `arg` (whether explicitly wrapper with a JIT decorator or not) may not
    use Numba's `with objmode` context.  Doing so will result in a runtime exception.

<p class="api-header">Parameters</p>

: __arg : *function, collections.abc.Mapping subclass or Series*:__ Mapping correspondence.  *function* may be a Python function or a dispatcher generated through numba.jit or bodo.jit.

: __na_actions : *{None, ‘ignore’}, default None*:__ If 'ignore' then NaN values will be propagated without passing them to the mapping correspondence.

: __engine : *{'bodo', 'python'}, default 'bodo'*:__  The engine to use to compute the UDF. By default, engine='bodo' will apply bodo.jit
to `arg` with fallback to python as described above. Use engine='python' to avoid any jit compilation.

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
