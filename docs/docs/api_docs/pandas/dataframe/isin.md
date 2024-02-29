# `pd.DataFrame.isin`


`pandas.DataFrame.isin(values)`

### Supported Arguments

- `values`: DataFrame (must have same indices) + iterable type, Numpy array types, Pandas array types, List/Tuple, Pandas Index Types (excluding interval Index and MultiIndex)

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   isin_1 = df.isin([1,5,9])
...   isin_2 = df.isin(pd.DataFrame({"A": [4,5,6], "C": [7,8,9]}))
...   formated_out = "\n".join([isin_1.to_string(), isin_2.to_string()])
...   return formated_out
>>> f()
      A      B      C
0  True   False  False
1  False  True   False
2  False  False  True
      A      B     C
0  False  False  True
1  False  False  True
2  False  False  True
```

!!! note

    `DataFrame.isin` ignores DataFrame indices. For example:

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
>>> f()
        A      B      C
        0  True  False  False
        1  True  False  False
        2  True  False  False

>>> def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
>>> f()
        A      B      C
        0  False  False  False
        1  False  False  False
        2  False  False  False
```

