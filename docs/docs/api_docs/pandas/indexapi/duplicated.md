# `pd.Index.duplicated`

`pandasIndex.duplicated(keep='first')`

### Supported Arguments: None

### Example Usage

```py
  
>>> @bodo.jit
... def f(I):
...   return I.duplicated()

>>> idx = pd.Index(['a', 'b', None, 'a', 'c', None, 'd', 'b'])
>>> f(idx)
array([False, False, False,  True, False,  True, False,  True])
```
