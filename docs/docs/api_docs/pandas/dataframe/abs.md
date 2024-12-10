# `pd.DataFrame.abs`

`pandas.DataFrame.abs()`

!!! note
Only supported for dataframes containing numerical data and Timedeltas

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,-2], "B": [3.1,-4.2], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(-10, unit="D")]})
...   return df.abs()
>>> f()
   A    B       C
0  1  3.1 10 days
1  2  4.2 10 days
```
