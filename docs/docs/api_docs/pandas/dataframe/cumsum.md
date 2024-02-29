# `pd.DataFrame.cumsum`


`pandas.DataFrame.cumsum(axis=None, skipna=True)`

### Supported Arguments : None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1, 2, 3], "B": [.1,np.NaN,12.3],})
...   return df.cumsum()
>>> f()
   A    B
0  1  0.1
1  3  NaN
2  6  NaN
```
!!! note
  Not supported for dataframe with nullable integer.


