# `pd.DataFrame.itertuples`


`pandas.DataFrame.itertuples(index=True, name='Pandas')`

### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   for x in pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}).itertuples():
...      print(x)
...      print(x[0])
...      print(x[2:])
>>> f()
Pandas(Index=0, A=1, B=4, C=7)
0
(4, 7)
Pandas(Index=1, A=2, B=5, C=8)
1
(5, 8)
Pandas(Index=2, A=3, B=6, C=9)
2
(6, 9)
```

