# `pd.core.window.rolling.Rolling.cov`

`pandas.core.window.rolling.Rolling.cov(other=None, pairwise=None, ddof=1)`

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
...   return df1.rolling(3).cov(df2)
A
0  NaN
1  NaN
2  1.0
3  1.0
4 -4.0
5 -5.0
6 -1.0
```

