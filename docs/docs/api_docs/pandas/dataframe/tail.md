# `pd.DataFrame.tail`

`pandas.DataFrame.tail(n=5)`

### Supported Arguments

- `n`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   return pd.DataFrame({"A": np.arange(1000)}).tail(3)
>>> f()
      A
997  997
998  998
999  999
```
