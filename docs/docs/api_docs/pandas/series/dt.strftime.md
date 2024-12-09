# `pd.Series.dt.strftime`

`pandas.Series.dt.strftime(date_format)`

### Supported Arguments

| argument      | datatypes | other requirements                                                                                                       |
|---------------|-----------|--------------------------------------------------------------------------------------------------------------------------|
| `date_format` | String    | Must be a valid [datetime format string](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.strftime("%B %d, %Y, %r")
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
>>> f(S)
0     January 01, 2022, 12:00:00 AM
1     January 01, 2022, 07:26:53 AM
2     January 01, 2022, 02:53:47 PM
3     January 01, 2022, 10:20:41 PM
4     January 02, 2022, 05:47:35 AM
5     January 02, 2022, 01:14:28 PM
6     January 02, 2022, 08:41:22 PM
7     January 03, 2022, 04:08:16 AM
8     January 03, 2022, 11:35:10 AM
9     January 03, 2022, 07:02:04 PM
10    January 04, 2022, 02:28:57 AM
11    January 04, 2022, 09:55:51 AM
12    January 04, 2022, 05:22:45 PM
13    January 05, 2022, 12:49:39 AM
14    January 05, 2022, 08:16:33 AM
15    January 05, 2022, 03:43:26 PM
16    January 05, 2022, 11:10:20 PM
17    January 06, 2022, 06:37:14 AM
18    January 06, 2022, 02:04:08 PM
19    January 06, 2022, 09:31:02 PM
20    January 07, 2022, 04:57:55 AM
21    January 07, 2022, 12:24:49 PM
22    January 07, 2022, 07:51:43 PM
23    January 08, 2022, 03:18:37 AM
24    January 08, 2022, 10:45:31 AM
25    January 08, 2022, 06:12:24 PM
26    January 09, 2022, 01:39:18 AM
27    January 09, 2022, 09:06:12 AM
28    January 09, 2022, 04:33:06 PM
29    January 10, 2022, 12:00:00 AM
dtype: object
```

