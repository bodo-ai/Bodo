# `pd.Series.str.encode`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.encode.html)

`pandas.Series.str.encode(encoding, errors='strict')`

### Argument Restrictions:
| argument 	  | datatypes   | other requirements  |
|-------------|-------------|---------------------|
| `encoding`  | String      | 					  |
| `errors`    | String      | 					  |

!!! note
	Input must be a Series of `String` data.

### Example Usage
``` py
>>> @bodo.jit
... def f(S):
...     return S.str.encode("ascii")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       b'A'
1      b'ce'
2      b'14'
3       b' '
4       b'@'
5     b'a n'
6    b'^ Ef'
dtype: large_binary[pyarrow]
```
