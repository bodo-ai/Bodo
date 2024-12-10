# `pd.Index.is_monotonic_increasing`

`pandasIndex.is_monotonic_increasing` and `pandas.Index.is_monotonic`

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
...   return I.is_monotonic_increasing

>>> I = pd.Index([1,2,3])
>>> f(I)
True

>>> @bodo.jit
... def g(I):
...   return I.is_monotonic

>>> I = pd.Index(1,2,3])
>>> g(I)
True
```
