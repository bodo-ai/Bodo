# `pd.Index.to_series`

`pandasIndex.to_series(index=None, name=None)`

### Supported Arguments:
  
  - `index`: can be a Index, Series, 1-dim numpy array, list, or tuple
  - `name`: can be a string or int

***Unsupported Index Types***

  - IntervalIndex
  - PeriodIndex
  - MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I, J):
...   return I.to_series(index=J)

>>> I = pd.Index([1, 4, 9, 0, 3])
>>> J = pd.Index(["A", "B", "C", "D", "E"])
>>> f(I, J)
A    1
B    4
C    9
D    0
E    3
dtype: int64
```

