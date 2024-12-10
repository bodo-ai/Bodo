# `pd.DataFrame.head`

`pandas.DataFrame.head(n=5)`

### Supported Arguments

- `head`: integer

### Example Usage

```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.DataFrame({"A": np.arange(1000)}).head(3)
       A
    0  0
    1  1
    2  2
```
