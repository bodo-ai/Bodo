# `pd.Index.nbytes`

`pandasIndex.nbytes`


***Unsupported Index Types***

  - MultiIndex
  - IntervalIndex

!!! info "Important"
    Currently, Bodo upcasts all numeric index data types to 64 bitwidth.

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.nbytes

>>> I1 = pd.Index([1,2,3,4,5,6], dtype = np.int64)
>>> f(I1)
48
>>> I2 = pd.Index([1,2,3], dtype = np.int64)
>>> f(I2)
24
>>> I3 = pd.Index([1,2,3], dtype = np.int32)
>>> f(I3)
24
```

