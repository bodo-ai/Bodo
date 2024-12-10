# `pd.Index.values`

`pandasIndex.values`

***Unsupported Index Types***

- MultiIndex
- IntervalIndex

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.values

>>> I = pd.Index([1,2,3])
>>> f(I)
[1 2 3]
```
