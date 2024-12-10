# `pd.core.window.rolling.Rolling.apply`

`pandas.core.window.rolling.apply(func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None)`

### Supported Arguments

- `func`: JIT function or callable defined within a JIT function
  - **Must be constant at Compile Time**
- `raw`: boolean
  - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df = pd.DataFrame({"A": [1,2,3,4,-5,-6,-7]})
...   return df.rolling(3).apply(lambda x: True if x.sum() > 0 else False)
A
0  NaN
1  NaN
2  1.0
3  1.0
4  1.0
5  0.0
6  0.0
```
