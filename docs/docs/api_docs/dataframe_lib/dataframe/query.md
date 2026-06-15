# bodo.pandas.BodoDataFrame.query
``` py
BodoDataFrame.query(expr, *, parser="pandas", engine=None, local_dict=None, global_dict=None, resolvers=None, level=0, inplace=False)
```
Query the columns (select the rows) of the BodoDataFrame with a boolean expression.

<p class="api-header">Parameters</p>

: __expr: *str*:__ The query string to evaluate.

: __local_dict *dict | None, default None*:__ A dictionary of local variables, taken from stack locals by default.

: __global_dict *dict | None, default None*:__ A dictionary of global variables, taken from stack globals by default.

: __resolvers: *list-like of dict-like | None, default None*:__ An additional list of maps between names and their respective values to use for variable lookup while parsing `expr`. The provided `resolvers` will be added to the resolvers used internally to inject the `BodoDataFrame.index` and `BodoDataFrame.columns` variables that refer to their respective BodoDataFrame instance attributes.

: __level: *int, default 0*:__ The number of prior stack frames to traverse and add to the current scope. Most users will *not* need to change this parameter (unless passing an unsupported argument; see below).

: Setting the `parser`, `engine`, or `inplace` parameters to a value other than the default will trigger a fallback to [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html). In such a case, if using variable references in `expr`, either the `level` parameter must be incremented or a `local_dict`/`global_dict` must be provided explicitly.

<p class="api-header">Returns</p>

: __BodoDataFrame__

<p class="api-header">Notes</p>

For more details on supported operations, functions, and formatting in the query string `expr`, see the pandas documentation at [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": [1, 3, 5, 7, 9],
        "B B": [0.6, 3.2, 0.19, 0.18, 7.4],
        "C&C": ["DD", "EEE", "F", "EE", "E"]
    }
)

bdf_queried1 = bdf.query("ceil(`B B`) == A or (A > `B B` and `C&C`.str.contains('EE'))")
print(bdf_queried1)
```

Output:
```
   A   B B C&C
0  1   0.6  DD
1  7  0.18  EE
```

---

``` py
list_var = [3.2, 0.18, 2, 10.1, 5]
bdf_queried2 = bdf.query("A in @list_var & not (`B B` == @list_var | `C&C`.str.len() > @length)", local_dict={"length": 1})
print(bdf_queried2)
```

Output:
```
   A   B B C&C
0  5  0.19   F
```