# pd.Series.empty

- `pandas.Series.empty`

### Example Usage

```py
>>> @bodo.jit
... def f(df):
...     return df.apply(lambda row: row.empty, axis=1)
>>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
>>> f(df)
0     False
1     False
2     False
3     False
4     False
      ...
95    False
96    False
97    False
98    False
99    False
Length: 100, dtype: boolean
```
