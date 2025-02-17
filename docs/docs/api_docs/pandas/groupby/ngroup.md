# `pd.core.groupby.DataFrameGroupby.ngroup`

`pandas.core.groupby.DataFrameGroupby.ngroup(ascending=True)`


### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("A").ngroup()
>>> df = pd.DataFrame({
...     "A": [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 1],
...     "B": np.arange(12.0)
... })
>>> f(df)
0     1
1     0
2     3
3     2
4     4
5     1
6     0
7     3
8     2
9     1
10    0
11    1
dtype: int64
```
  
