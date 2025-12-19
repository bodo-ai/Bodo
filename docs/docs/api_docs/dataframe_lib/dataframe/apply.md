
# bodo.pandas.BodoDataFrame.apply
``` py
BodoDataFrame.apply(
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="bodo",
        engine_kwargs=None,
        **kwargs,
    ) -> BodoSeries
```

Apply a function along an axis of the BodoDataFrame.

Currently only supports applying a function that returns a scalar value for each row (i.e. `axis=1`).
All other uses will fall back to Pandas.
See [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) for more details.

By default, bodo.jit will be applied to `func`.  If this JIT compilation fails for any
reason, the mapping function will be run as a normal Python function.  If the compilation succeeds,
the JIT compiled function will be used for apply and the overheads associated with running Python code
from within the execution pipeline are avoided.

!!! note
    Calling `BodoDataFrame.apply` will immediately execute a plan if this JIT compilation fails,
    generating a small sample of the BodoDataFrame and calling `pandas.DataFrame.apply` on the sample to
    infer output types before proceeding with lazy evaluation.

!!! note
    Functions passed to `func` (whether explicitly wrapper with a JIT decorator or not) may not
    use Numba's `with objmode` context.  Doing so will result in a runtime exception.

<p class="api-header">Parameters</p>

: __func : *function*:__ Function to apply to each row.

: __axis : *{0 or 1}, default 0*:__ The axis to apply the function over. `axis=0` will fall back to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply).

: __args : *tuple*:__ Additional positional arguments to pass to *func*.

: __engine : *{'bodo', 'python', 'numba'}, default 'bodo'*:__ The engine to use to compute the UDF. By default, `engine="bodo"` will apply bodo.jit
to `func` with fallback to Python described above. Use engine='python' to avoid any jit compilation. `engine='numba'` will trigger a fall back to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply).

: __\*\*kwargs:__ Additional keyword arguments to pass as keyword arguments to *func*.


: All other parameters will trigger a fallback to [`pandas.DataFrame.apply`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html#pandas.DataFrame.apply) if a non-default value is provided.

<p class="api-header">Returns</p>
: __BodoSeries:__ The result of applying *func* to each row in the BodoDataFrame.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
        {
            "a": bd.array([1, 2, 3] * 4, "Int64"),
            "b": bd.array([4, 5, 6] * 4, "Int64"),
            "c": ["a", "b", "c"] * 4,
        },
    )

out_bodo = bdf.apply(lambda x: x["a"] + 1, axis=1)

print(type(out_bodo))
print(out_bodo)
```

Output:
```
<class 'bodo.pandas.series.BodoSeries'>
0     2
1     3
2     4
3     2
4     3
5     4
6     2
7     3
8     4
9     2
10    3
11    4
dtype: int64[pyarrow]
```
---