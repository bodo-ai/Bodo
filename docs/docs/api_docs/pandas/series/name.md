# pd.Series.name

-   `pandas.Series.name`

### Example Usage

``` py
>>> @bodo.jit
... def f(df):
...     return df.apply(lambda row: row.name, axis=1)
>>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
>>> f(df)
0      0
1      1
2      2
3      3
4      4
      ..
95    95
96    96
97    97
98    98
99    99
Length: 100, dtype: int64
```
