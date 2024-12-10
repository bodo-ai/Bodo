# `pd.Index.is_all_dates`

`pandasIndex.is_all_dates`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.is_all_dates

>>> I = pd.date_range("2018-01-01", "2018-01-06")
>>> f(I)
True
```
