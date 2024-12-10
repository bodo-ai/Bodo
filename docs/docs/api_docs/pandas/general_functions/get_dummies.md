# `pd.get_dummies`

`pandas.get_dummies(data, prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)`

### Supported Arguments

| argument            | datatypes                                 | other requirements                              |
|---------------------|-------------------------------------------|-------------------------------------------------|
| `data`              | Array or Series with Categorical dtypes   |  **Categories must be  known at compile time.** |   


### Example Usage

```py

>>> @bodo.jit
... def f(S):
...     return pd.get_dummies(S)

>>> S = pd.Series(["CC", "AA", "B", "D", "AA", None, "B", "CC"]).astype("category")
>>> f(S)

AA  B  CC  D
0   0  0   1  0
1   1  0   0  0
2   0  1   0  0
3   0  0   0  1
4   1  0   0  0
5   0  0   0  0
6   0  1   0  0
7   0  0   1  0
```
