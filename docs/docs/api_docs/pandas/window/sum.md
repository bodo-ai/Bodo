# `pd.core.window.rolling.Rolling.sum`

`pandas.core.window.rolling.Rolling.sum(engine=None, engine_kwargs=None)`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
...   return df.rolling(3).sum()
    A     B
0   NaN   NaN
1   NaN   NaN
2   6.0  27.0
3   9.0   NaN
4  12.0   NaN
5  15.0   NaN
6  18.0  36.0
```

