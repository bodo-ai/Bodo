# `pd.DataFrame.sort_values`

`pandas.DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)`

### Supported Arguments

- `by`: constant String or constant list of strings
- `ascending`: boolean, list/tuple of boolean, with length equal to the number of key columns
- `inplace`: Constant boolean
- `na_position`: constant String ("first" or "last"), constant list/tuple of String, with length equal to the number of key columns

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,2,None], "B": [4, 5, 6, None]})
...   df.sort_values(by=["A", "B"], ascending=[True, False], na_position=["first", "last"], inplace=True)
...   return df
>>> f()
      A     B
3  <NA>  <NA>
0     1     4
2     2     6
1     2     5
```
