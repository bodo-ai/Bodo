# `pd.DataFrame.info`

`pandas.DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None)`

### Supported Arguments: None

### Example Usage

```
>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
...   return df.info()
>>> f()
<class 'DataFrameType'>
RangeIndexType(none): 3 entries, 0 to 2
Data columns (total 3 columns):
#   Column  Non-Null Count  Dtype

0  A       3 non-null      int64
1  B       3 non-null      unicode_type
2  C       3 non-null      timedelta64[ns]
dtypes: int64(1), timedelta64[ns](1), unicode_type(1)
memory usage: 108.0 bytes

```

!!! note
The exact output string may vary slightly from Pandas.
