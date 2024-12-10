# `pd.DataFrame.filter`

`pandas.DataFrame.filter(items=None, like=None, regex=None, axis=None)`


### Supported Arguments


- `items`: Constant list of String
- `like`: Constant string
- `regex`: Constant String
- `axis` (only supports the "column" axis): Constant String, Constant integer


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]})
...   filtered_df_1 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(items = ["A"])
...   filtered_df_2 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(like ="hello", axis = "columns")
...   filtered_df_3 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(regex="(ab){3}", axis = 1)
...   formated_out = "\n".join([filtered_df_1.to_string(), filtered_df_2.to_string(), filtered_df_3.to_string()])
...   return formated_out
>>> f()
   A
0  3
  hello world
0            2
  ababab
0       1
```

