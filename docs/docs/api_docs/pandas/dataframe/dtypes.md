# `pd.DataFrame.dtypes`

`pandas.DataFrame.dtypes`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
...   return df.dtypes
>>> f()
A              int64
B             string
C    timedelta64[ns]
dtype: object
```
