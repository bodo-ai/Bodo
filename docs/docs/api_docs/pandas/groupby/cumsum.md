# `pd.core.groupby.Groupby.cumsum`

`pandas.core.groupby.Groupby.cumsum(axis=0)`

!!! note
    `cumsum` is only supported on numeric columns and is not supported on boolean columns

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").cumsum()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

        A            C
0     1.0        1.510
1     2.0        2.421
2    25.0   233233.510
3     NaN       14.631
4    26.0   233235.020
5     4.0       17.052
6    50.0   466467.020
7     NaN       29.262
8    51.0   466468.530
9     6.0       31.683
10   75.0   699700.530
11    NaN       43.893
12   76.0   699702.040
13    8.0       46.314
14  100.0   932934.040
15    NaN       58.524
16  101.0   932935.550
17   10.0       60.945
18  125.0  1166167.550
19    NaN       73.155
```

