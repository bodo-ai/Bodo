# `pd.DataFrame.sort_index`


`pandas.DataFrame.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)`


### Supported Arguments

- `ascending`: boolean
- `na_position`:constant String ("first" or "last")


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3]}, index=[1,None,3])
...   return df.sort_index(ascending=False, na_position="last")
>>> f()
     A
3    3
1    1
NaN  2
```

