# `pd.DataFrame.replace`

`pandas.DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')`

### Supported Arguments

- `to_replace`: various scalars
  - **Required argument**
- `value`: various scalars
  - Must be of the same type as to_replace

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.replace(1, -1)
>>> f()
   A  B  C
0 -1  4  7
1  2  5  8
2  3  6  9
```
