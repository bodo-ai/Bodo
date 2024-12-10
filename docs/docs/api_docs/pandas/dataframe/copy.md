# `pd.DataFrame.copy`

`pandas.DataFrame.copy(deep=True)`


### Supported Arguments

- `copy`: boolean


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3]})
...   shallow_df = df.copy(deep=False)
...   deep_df = df.copy()
...   shallow_df["A"][0] = -1
...   formated_out = "\n".join([df.to_string(), shallow_df.to_string(), deep_df.to_string()])
...   return formated_out
>>> f()
   A
0  -1
1  2
2  3
  A
0  -1
1  2
2  3
  A
0  1
1  2
2  3
```

