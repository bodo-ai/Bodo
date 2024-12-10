# `pd.Index.rename`

`pandasIndex.rename(name, inplace=False)`

### Supported Arguments

- `name`: label or list of labels

***Unsupported Index Types***

- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I, name):
...   return I.rename(name)

>>> I = pd.Index(["a", "b", "c"])
>>> f(I, "new_name")
Index(['a', 'b', 'c'], dtype='object', name='new_name')
```
