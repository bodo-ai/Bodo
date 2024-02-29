# `pd.DataFrame.mask`

`pandas.DataFrame.mask(cond, other=np.nan, inplace=False, axis=1, level=None, errors='raise', try_cast=NoDefault.no_default)`


### Supported Arguments

- `cond`: Boolean DataFrame,Boolean Series,Boolean Array
   - If 1-dimensional array or Series is provided, equivalent to Pandas `df.mask` with `axis=1`.
- `other`: Scalar, DataFrame, Series, 1 or 2-D Array
   - `None`, - Data types in `other` must match corresponding entries in DataFrame.
   - `None` or omitting argument defaults to the respective `NA` value for each type.

!!! note
    DataFrame can contain categorical data if `other` is a scalar.

### Example Usage

```py

>>> @bodo.jit
... def f(df, cond, other):
...   return df.mask(cond, other)
>>> df = pd.DataFrame({"A": [1,2,3], "B": [4.3, 2.4, 1.2]})
>>> cond = df > 2
>>> other = df + 100
>>> f(df, cond, other)
    A      B
0    1  104.3
1    2  102.4
2  103    1.2
```
