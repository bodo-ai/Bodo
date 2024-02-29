# `pd.DataFrame.query`


-  pandas.DataFrame.query(expr, inplace=False, \**kwargs)


### Supported Arguments

- `expr`:  Constant String

### Example Usage

```py

>>> @bodo.jit
... def f(a):
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.query('A > @a')
>>> f(1)
   A  B  C
1  2  5  8
2  3  6  9
```

!!! note
    * The output of the query must evaluate to a 1d boolean array.
    * Cannot refer to the index by name in the query string.
    * Query must be one line.
    * If using environment variables, they should be passed as arguments to the function.

