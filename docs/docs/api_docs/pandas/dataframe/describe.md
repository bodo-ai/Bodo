# `pd.DataFrame.describe`


`pandas.DataFrame.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)`

### Supported Arguments : None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
...   return df.describe()
>>> f()
        A                    B
count  3.0                    3
mean   2.0  2001-07-16 16:00:00
min    1.0  2000-10-02 00:00:00
25%    1.5  2001-03-20 00:00:00
50%    2.0  2001-09-05 00:00:00
75%    2.5  2001-12-07 12:00:00
max    3.0  2002-03-11 00:00:00
std    1.0                  NaN
```

!!! note
    Only supported for dataframes containing numeric data, and datetime data. Datetime_is_numeric defaults to True in JIT code.

