# `pd.DataFrame.select_dtypes`

`pandas.DataFrame.select_dtypes(include=None, exclude=None)`

### Supported Arguments


- `include`: string, type, List or tuple of string/type
    - **Must be constant at Compile Time**
- `exclude`: string, type, List or tuple of string/type
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df= pd.DataFrame({"A": [1], "B": ["X"], "C": [pd.Timedelta(10, unit="D")], "D": [True], "E": [3.1]})
...   out_1 = df_l.select_dtypes(exclude=[np.float64, "bool"])
...   out_2 = df_l.select_dtypes(include="int")
...   out_3 = df_l.select_dtypes(include=np.bool_, exclude=(np.int64, "timedelta64[ns]"))
...   formated_out = "\n".join([out_1.to_string(), out_2.to_string(), out_3.to_string()])
...   return formated_out
>>> f()
   A  B       C
0  1  X 10 days
  A
0  1
      D
0  True
```

