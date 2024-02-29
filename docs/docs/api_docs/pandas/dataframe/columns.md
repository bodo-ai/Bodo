# `pd.DataFrame.columns`

`pandas.DataFrame.columns`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
...   return df.columns
>>> f()
Index(['A', 'B', 'C'], dtype='object')
```

