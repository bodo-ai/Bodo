# `pd.core.window.rolling.Rolling.mean`

`pandas.core.window.rolling.Rolling.mean(engine=None, engine_kwargs=None)`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
...   return df.rolling(3).mean()
  A     B
0  NaN   NaN
1  NaN   NaN
2  2.0   9.0
3  3.0   NaN
4  4.0   NaN
5  5.0   NaN
6  6.0  12.0
```

