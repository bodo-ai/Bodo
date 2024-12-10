# `pd.Series.iat`

`pandas.Series.iat`

We only support indexing using `iat` using a pair of integers

### Example Usage
>
``` py
>>> @bodo.jit
... def f(S, i):
...   return S.iat[i]
>>> S = pd.Series(np.arange(1000))
>>> f(S, 27)
27
```

