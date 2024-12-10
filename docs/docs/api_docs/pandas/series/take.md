# `pd.Series.take`

`pandas.Series.take(indices, axis=0, is_copy=None)`

### Supported Arguments

| argument | datatypes | other requirements |
|-----------|------------------------------|---------------------------------------------------------------------------------------|
| `indices` | Array like with integer data | To have distributed data `indices` must be an array with the same distribution as S. |

!!! note
Bodo does not accept any additional arguments for Numpy
compatibility

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.take([2, 7, 4, 19])
>>> S = pd.Series(np.arange(100))
>>> f(S)
2      2
7      7
4      4
19    19
dtype: int64
```
