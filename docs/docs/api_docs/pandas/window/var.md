# `pd.core.window.rolling.Rolling.var`

`pandas.core.window.rolling.Rolling.var(ddof=1)`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
...   return df.rolling(3).var()
  A    B
0  NaN  NaN
1  NaN  NaN
2  1.0  1.0
3  1.0  NaN
4  1.0  NaN
5  1.0  NaN
6  1.0  1.0
```

