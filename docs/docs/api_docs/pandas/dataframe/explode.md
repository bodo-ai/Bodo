# `pd.DataFrame.explode`


`pandas.DataFrame.explode(column, ignore_index=False)`


### Supported Arguments

- `column`: Constant Column label or list of labels

### Example Usage

```py
>>> @bodo.jit
... def f(df, cols):
...   return df.explode(cols)
>>> df = pd.DataFrame({"A": [[0, 1, 2], [5], [], [3, 4]], "B": [1, 7, 2, 4], "C": [[1, 2, 3], np.nan, [], [1, 2]]})
>>> f(df, ["A", "C"])
      A  B     C
0     0  1     1
0     1  1     2
0     2  1     3
1     5  7  <NA>
2  <NA>  2  <NA>
3     3  4     1
3     4  4     2
```

