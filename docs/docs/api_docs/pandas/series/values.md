# pd.Series.values

-   `pandas.Series.values`

### Example Usage

``` py
>>> @bodo.jit
... def f(df):
...     return df.apply(lambda row: row.values, axis=1)
>>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
>>> f(df)
0      (0, A)
1      (1, b)
2      (2, A)
3      (3, b)
4      (4, A)
      ...
95    (95, b)
96    (96, A)
97    (97, b)
98    (98, A)
99    (99, b)
Length: 100, dtype: object
```

