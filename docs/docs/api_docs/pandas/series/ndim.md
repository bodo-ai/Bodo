# pd.Series.ndim

-   `pandas.Series.ndim`

### Example Usage

``` py
>>> @bodo.jit
... def f(df):
...     return df.apply(lambda row: row.ndim, axis=1)
>>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
>>> f(df)
0     1
1     1
2     1
3     1
4     1
..
95    1
96    1
97    1
98    1
99    1
Length: 100, dtype: int64
```

