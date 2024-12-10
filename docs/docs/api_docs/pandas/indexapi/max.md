# `pd.Index.max`

`pandasIndex.max(axis=None, skipna=True, *args, **kwargs)`

### Supported Arguments: None

***Supported Index Types***

- NumericIndex
- RangeIndex
- CategoricalIndex
- TimedeltaIndex
- DatetimeIndex

!!! info "Important"
\- Bodo **Does Not** support `args` and `kwargs`, even for compatibility.
\- For DatetimeIndex, will throw an error if all values in the index are null.

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.min()

>>> I = pd.Index(pd.date_range(start="2018-04-24", end="2018-04-25", periods=5))
>>> f(I)
2018-04-25 00:00:00
```
