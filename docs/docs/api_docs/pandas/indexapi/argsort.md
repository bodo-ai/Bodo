# `pd.Index.argsort`

`pandasIndex.argsort(*args, **kwargs)`


### Supported Arguments: None

***Unsupported Index Types***

- IntervalIndex
- MultiIndex


### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.argsort()

>>> I = pd.Index(["A", "L", "P", "H", "A"])
>>> f(I)
array([0, 4, 3, 1, 2])
```

