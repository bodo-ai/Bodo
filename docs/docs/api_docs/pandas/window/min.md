# `pd.core.window.rolling.Rolling.min`

`pandas.core.window.rolling.Rolling.min(engine=None, engine_kwargs=None)`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
...   return df.rolling(3).min()
  A     B
0  NaN   NaN
1  NaN   NaN
2  1.0   8.0
3  2.0   NaN
4  3.0   NaN
5  4.0   NaN
6  5.0  11.0
```

