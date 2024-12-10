# pd.Series.shape

- `pandas.Series.shape`

### Example Usage

```py
>>> @bodo.jit
... def f(df):
...     return df.apply(lambda row: row.shape, axis=1)
>>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
>>> f(df)
0     (2,)
1     (2,)
2     (2,)
3     (2,)
4     (2,)
      ...
95    (2,)
96    (2,)
97    (2,)
98    (2,)
99    (2,)
Length: 100, dtype: object
```
