# `pd.Series.cat.codes`

`pandas.Series.cat.codes`

!!! note
If categories cannot be determined at compile time, then Bodo
defaults to creating codes with an `int64`, which may differ from
Pandas.

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.cat.codes
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@"] * 10).astype("category")
>>> f(S)
0     2
1     4
2     1
3     3
4     0
5     2
6     4
7     1
8     3
9     0
10    2
11    4
12    1
13    3
14    0
15    2
16    4
17    1
18    3
19    0
20    2
21    4
22    1
23    3
24    0
25    2
26    4
27    1
28    3
29    0
30    2
31    4
32    1
33    3
34    0
35    2
36    4
37    1
38    3
39    0
40    2
41    4
42    1
43    3
44    0
45    2
46    4
47    1
48    3
49    0
dtype: int8
```

### Serialization / IO / Conversion
