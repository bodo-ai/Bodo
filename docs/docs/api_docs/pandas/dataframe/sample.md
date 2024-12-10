# `pd.DataFrame.sample`


`pandas.DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None, ignore_index=False)`


### Supported Arguments

- `n`: Integer
- `frac`: Float
- `replace`: boolean


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.sample(1)
>>> f()
   A  B  C
2  3  6  9
```

