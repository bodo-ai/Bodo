
# `pd.crosstab`

`pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)`

### Supported Arguments

 | argument  | datatypes  |
 |-----------|------------|
 | `index`   | SeriesType |
 | `columns` | SeriesType |
 

!!! note
    Annotation of pivot values is required. For example,
    `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
    the output table `pt` will have columns called `small` and `large`.

### Example Usage

```py

>>> @bodo.jit(pivots={"pt": ["small", "large"]})
... def f(df):
...   pt = pd.crosstab(df.A, df.C)
...   return pt

>>> list_A = ["foo", "foo", "bar", "bar", "bar", "bar"]
>>> list_C = ["small", "small", "large", "small", "small", "middle"]
>>> df = pd.DataFrame({"A": list_A, "C": list_C})
>>> f(df)

         small  large
index
foo          2      0
bar          2      1
```
