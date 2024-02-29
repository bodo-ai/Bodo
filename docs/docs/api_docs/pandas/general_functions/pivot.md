# `pd.pivot`

 `pandas.pivot(data, values=None, index=None, columns=None)`


### Supported Arguments

| argument                    | datatypes                                  |
|-----------------------------|--------------------------------------------|
| `data`                      |   DataFrame                                |
| `values`                    |   Constant Column Label or list of  labels |
| `index`                     |   Constant Column Label or list of  labels |
| `columns`                   |   Constant Column Label                    |

!!! note
    The the number of columns and names of the output DataFrame won't be known
    at compile time. To update typing information on DataFrame you should pass it back to Python.


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": ["X","X","X","X","Y","Y"], "B": [1,2,3,4,5,6], "C": [10,11,12,20,21,22]})
...   pivoted_tbl = pd.pivot(data, columns="A", index="B", values="C")
...   return pivoted_tbl
>>> f()
A     X     Y
B
1  10.0   NaN
2  11.0   NaN
3  12.0   NaN
4  20.0   NaN
5   NaN  21.0
6   NaN  22.0
```
