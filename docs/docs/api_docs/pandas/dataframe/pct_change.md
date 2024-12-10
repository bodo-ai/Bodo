# `pd.DataFrame.pct_change`

`pandas.DataFrame.pct_change(periods=1, fill_method='pad', limit=None, freq=None)`

### Supported Arguments

- `periods`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [10,100,1000,10000]})
...   return df.pct_change()
>>> f()
    A
0  NaN
1  9.0
2  9.0
3  9.0
```
