# `pd.Index.drop_duplicates`

`pandasIndex.drop_duplicates(keep='first')`

### Supported Arguments: None

***Unsupported Index Types***

- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.drop_duplicates()

>>> I = pd.Index(["a", "b", "c", "a", "b", "c"])
>>> f(I)
Index(['a', 'b', 'c'], dtype='object')
```
