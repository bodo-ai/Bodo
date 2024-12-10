# `pd.notnull`

`pandas.notnull(obj)`

### Supported Arguments

| argument | datatypes |
|----------|--------------------------------------------|
| `obj` | DataFrame, Series, Index, Array, or Scalar |

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return pd.notnull(df)

>>> df = pd.DataFrame(
...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
...    [1.1, -2.1, 7.1, 0.1, 3.1],
... )
>>> f(df)

       A     B
1.1   True  True
-2.1  False  True
7.1   True  True
0.1   True  True
3.1   True  True

```
