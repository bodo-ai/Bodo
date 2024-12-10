# `pd.core.window.rolling.Rolling.corr`

`pandas.core.window.rolling.Rolling.corr(other=None, pairwise=None, ddof=1)`

### Supported Arguments

- `other`: DataFrame or Series (cannot contain nullable Integer Types)
- **Required**
  - If called with a DataFrame, `other` must be a DataFrame. If called with a Series, `other` must be a Series.

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   df1 = pd.DataFrame({"A": [1,2,3,4,5,6,7]})
...   df2 = pd.DataFrame({"A": [1,2,3,4,-5,-6,-7]})
...   return df1.rolling(3).corr(df2)
        A
0       NaN
1       NaN
2  1.000000
3  1.000000
4 -0.810885
5 -0.907841
6 -1.000000
```
