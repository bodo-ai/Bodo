# `pd.core.window.rolling.Rolling.count`


`pandas.core.window.rolling.Rolling.count()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,5], "B": [6,7,None,9,10]})
...   return df.rolling(3).count()
A    B
0  1.0  1.0
1  2.0  2.0
2  3.0  3.0
3  3.0  2.0
4  3.0  2.0
5  3.0  2.0
6  3.0  3.0
```

