# `pd.DataFrame.shift`


`pandas.DataFrame.shift(periods=1, freq=None, axis=0, fill_value=NoDefault.no_default)`


### Supported Arguments

- `periods`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
...   return df.shift(1)
>>> f()
     A    B
0  NaN  NaN
1  1.0  4.0
2  1.0  5.0
```

!!! note
    Only supported for dataframes containing numeric, boolean, datetime.date and string types.



