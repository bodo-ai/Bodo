# `pd.DataFrame.quantile`

`pandas.DataFrame.quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')`

### Supported Arguments

- `q`: Float or Int
  - must be 0\<= q \<= 1
- `axis`: Integer (0 or 1)
  - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.quantile()
>>> f()
A    2.0
B    5.0
C    8.0
dtype: float64
dtype: int64
```
