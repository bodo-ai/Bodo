# `pd.core.window.rolling.Rolling.max`

`pandas.core.window.rolling.Rolling.max(engine=None, engine_kwargs=None)`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
...   return df.rolling(3).max()
  A     B
0  NaN   NaN
1  NaN   NaN
2  3.0  10.0
3  4.0   NaN
4  5.0   NaN
5  6.0   NaN
6  7.0  13.0
```
