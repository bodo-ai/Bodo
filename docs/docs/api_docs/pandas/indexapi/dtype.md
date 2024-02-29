# `pd.Index.dtype`


`pandasIndex.dtype`


***Unsupported Index Types***

  - PeriodIndex
  - IntervalIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.dtype

>>> I = pd.Index([1,2,3,4])
>>> f(I)
dtype('int64')
``` 


