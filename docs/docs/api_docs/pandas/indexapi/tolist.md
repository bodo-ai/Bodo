# `pd.Index.tolist`

`pandasIndex.tolist()`

***Unsupported Index Types***

- PeriodIndex
- IntervalIndex
- DatetimeIndex
- TimedeltaIndex
- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.tolist()

>>> I = pd.RangeIndex(5, -1, -1)
>>> f(I)
[5, 4, 3, 2, 1, 0]
```

## Numeric Index

Numeric index objects `RangeIndex`, `Int64Index`, `UInt64Index` and
`Float64Index` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.
