# `pd.DataFrame.iloc`

`pandas.DataFrame.iloc`

### getitem

- `df.iloc` supports single integer indexing (returns row as series) `df.iloc[0]`
- `df.iloc` supports single list/array/series of integers/bool `df.iloc[[0,1,2]]`
- for tuples indexing `df.iloc[row_idx, col_idx]` we allow:
  - `row_idx` to be int list/array/series of integers/bool slice
  - `col_idx` to be constant int, constant list of integers, or constant slice
- e.g.: `df.iloc[[0,1,2], :]`

### setitem

- `df.iloc` only supports scalar setitem
- `df.iloc` only supports tuple indexing `df.iloc[row_idx, col_idx]`
- `row_idx` can be anything supported for series setitem:
  - int
  - list/array/series of integers/bool
  - slice
- `col_idx` can be: constant int, constant list/tuple of integers

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   df.iloc[0, 0] = df.iloc[2,2]
...   df.iloc[1, [1,2]] = df.iloc[0, 1]
...   df["D"] = df.iloc[0]
...   return df
>>> f()
   A  B  C  D
0  9  4  7  7
1  2  4  4  4
2  3  6  9  9
```
