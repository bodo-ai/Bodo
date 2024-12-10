# `pd.DataFrame.merge`


`pandas.DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)`


!!! note
    See [`pd.merge`][pdmerge] for full list of supported arguments, and more examples.

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
...   return df.merge(pd.DataFrame({"C": [-1,-2,-3], "D": [4,4,6]}), left_on = "B", right_on = "D")
>>> f()
   A  B  C  D
0  1  4 -1  4
1  1  4 -2  4
2  3  6 -3  6
```





