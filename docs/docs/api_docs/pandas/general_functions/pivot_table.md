# `pd.pivot_table`


`pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)`


### Supported Arguments

| argument  | datatypes                                |
|-----------|------------------------------------------|
| `data`    | DataFrame                                |
| `values`  | Constant Column Label or list of  labels |
| `index`   | Constant Column Label or list of  labels |
| `columns` | Constant Column Label                    |
| `aggfunc` | String Constant                          |


!!! note
    This code takes two different paths depending on if pivot values are annotated. When
    pivot values are annotated then output columns are set to the annotated values.
    For example, `@bodo.jit(pivots={'pt': ['small', 'large']})`
    declares the output pivot table `pt` will have columns called `small` and `large`.

    If pivot values are not annotated, then the number of columns and names of the output DataFrame won't be known
    at compile time. To update typing information on DataFrame you should pass it back to Python.


### Example Usage

```py

>>> @bodo.jit(pivots={'pivoted_tbl': ['X', 'Y']})
... def f():
...   df = pd.DataFrame({"A": ["X","X","X","X","Y","Y"], "B": [1,2,3,4,5,6], "C": [10,11,12,20,21,22]})
...   pivoted_tbl = pd.pivot_table(df, columns="A", index="B", values="C", aggfunc="mean")
...   return pivoted_tbl
>>> f()
      X     Y
B
1  10.0   NaN
2  11.0   NaN
3  12.0   NaN
4  20.0   NaN
5   NaN  21.0
6   NaN  22.0
```
