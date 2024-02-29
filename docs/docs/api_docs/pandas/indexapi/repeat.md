# `pd.Index.repeat`

`pandasIndex.repeat(repeats, axis=None)`

### Supported Arguments:

  - `repeat`: can be a non-negative integer or array of non-negative integers

***Supported Index Types***

  - NumericIndex
  - StringIndex
  - RangeIndex
  - DatetimeIndex
  - TimedeltaIndex
  - CategoricalIndex

!!! info "Important"
      If repeats is an integer array but its size is not the same as the length of I, undefined behavior may occur.

### Example Usage

```py
>>> @bodo.jit(distributed=["I"])
... def f(I):
...    return I.repeat(3)

>>> I = pd.Index(["A", "B", "C"])
>>> f(I)
Index(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], dtype='object')
```

### Missing values


