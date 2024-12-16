# `pd.Series.tolist`

`pandas.Series.tolist()`


!!! note
    Calling `tolist` on a non-float array with NA values with cause a
    runtime exception.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.tolist()
>>> S = pd.Series(np.arange(50))
>>> f(S)
[0,
 1,
 2,
 3,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 11,
 12,
 13,
 14,
 15,
 16,
 17,
 18,
 19,
 20,
 21,
 22,
 23,
 24,
 25,
 26,
 27,
 28,
 29,
 30,
 31,
 32,
 33,
 34,
 35,
 36,
 37,
 38,
 39,
 40,
 41,
 42,
 43,
 44,
 45,
 46,
 47,
 48,
 49]
```

### Indexing, iteration:

Location based indexing using `[]`, `iat`, and
`iloc` is supported. Changing values of existing string
Series using these operators is not supported yet.

