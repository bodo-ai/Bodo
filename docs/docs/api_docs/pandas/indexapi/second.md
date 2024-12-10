# `pd.DateTimeIndex.second`


`pandasDatetimeIndex.second`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.second

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([45, 35, 25, 15, 5], dtype='int64')
```


