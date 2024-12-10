# `pd.DataFrame.diff`


`pandas.DataFrame.diff(periods=1, axis=0)`


### Supported Arguments

- `periods`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
...   return df.diff(1)
>>> f()
     A        B
0  NaN      NaT
1  1.0 338 days
2  1.0 187 days
```
!!! note
    Only supported for dataframes containing float, non-null int, and datetime64ns values


