# `pd.Index.is_monotonic_decreasing`


`pandasIndex.is_monotonic_decreasing`


***Unsupported Index Types***

  - StringIndex
  - BinaryIndex
  - IntervalIndex
  - CategoricalIndex
  - PeriodIndex
  - MultiIndex


### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.is_monotonic_decreasing

>>> I = pd.Index([1,2,3])
>>> f(I)
False
```


