# `pd.DataFrame.to_string`

- `pandas.DataFrame.to_string(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, min_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None, max_colwidth=None, encoding=None)`

### Supported Arguments

- `buf`
- `columns`
- `col_space`
- `header`
- `index`
- `na_rep`
- `formatters`
- `float_format`
- `sparsify`
- `index_names`
- `justify`
- `max_rows`
- `min_rows`
- `max_cols`
- `how_dimensions`
- `decimal`
- `line_width`
- `max_colwidth`
- `encoding`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3]})
...   return df.to_string()
>>> f()
   A
0  1
1  2
2  3
```

!!! note
\* This function is not optimized.
\* When called on a distributed dataframe, the string returned for each rank will be reflective of the dataframe for that rank.
