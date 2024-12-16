# `pd.to_numeric`


`pandas.to_numeric(arg, errors="raise", downcast=None)`

### Supported Arguments
   
| argument   | datatypes                                                            | other requirements                                       |
|------------|----------------------------------------------------------------------|----------------------------------------------------------|
| `arg`      | Series or Array                                                      |                                                          |
| `downcast` | String and one of (`'integer'`, `'signed'`, `'unsigned'`, `'float'`) | <ul><li> **Must be constant at Compile Time** </li></ul> |

!!! note

    * Output type is float64 by default
    * Unlike Pandas, Bodo does not dynamically determine output type,
      and does not downcast to the smallest numerical type.
    * `downcast` parameter should be used for type annotation of output.

### Example Usage

```py

>>> @bodo.jit
... def f(S):
...     return pd.to_numeric(S, errors="coerce", downcast="integer")

>>> S = pd.Series(["1", "3", "12", "4", None, "-555"])
>>> f(S)

0       1
1       3
2      12
3       4
4    <NA>
5    -555
dtype: Int64
```
