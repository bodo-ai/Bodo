# `pd.Series.idxmax`

`pandas.Series.idxmax(axis=0, skipna=True)`

### Supported Arguments None

!!! note
Bodo does not accept any additional arguments for Numpy
compatibility

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.idxmax()
>>> S = pd.Series(np.arange(100))
>>> S[(S % 3 == 0)] = 100
>>> f(S)
0
```
