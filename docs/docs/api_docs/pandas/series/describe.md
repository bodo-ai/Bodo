# `pd.Series.describe`

`pandas.Series.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)`

### Supported Arguments None

!!! note
    Bodo only supports numeric and datetime64 types and assumes
    `datetime_is_numeric=True`


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.describe()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
count    100.000000
mean       2.950000
std        2.021975
min        0.000000
25%        1.000000
50%        3.000000
75%        5.000000
max        6.000000
dtype: float64
```

