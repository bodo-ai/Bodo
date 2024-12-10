# `pd.isna`

`pandas.isna(obj)`

### Supported Arguments

| argument | datatypes |
|-------------------------------------|--------------------------------------------|
| `obj` | DataFrame, Series, Index, Array, or Scalar |

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return pd.isna(df)

>>> df = pd.DataFrame(
...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
...    [1.1, -2.1, 7.1, 0.1, 3.1],
... )
>>> f(df)

       A      B
1.1  False  False
-2.1   True  False
7.1  False  False
0.1  False  False
3.1  False  False
```
