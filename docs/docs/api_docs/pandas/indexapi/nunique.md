# `pd.Index.nunique`

`pandasIndex.nunique(dropna=True)`

### Supported Arguments:

- `dropna`: can be True or False

***Unsupported Index Types***

- IntervalIndex
- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.nunique()

>>> I = pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1])
>>> f(I)
4
```
