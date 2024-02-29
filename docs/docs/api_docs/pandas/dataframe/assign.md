# `pd.DataFrame.assign`

`pandas.DataFrame.assign(\**kwargs)`

### Example Usage

```py
>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
...   df2 = df.assign(C = 2 * df["B"], D = lambda x: x.C -1)
...   return df2
>>> f()
   A  B   C   D
0  1  4   8  -8
1  2  5  10 -10
2  3  6  12 -12
```

!!! note
    arguments can be JIT functions, lambda functions, or values that can be used to initialize a Pandas Series.


