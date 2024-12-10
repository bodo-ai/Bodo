# TO_CHAR

- `#!sql TO_CHAR(COLUMN_EXPRESSION)`

Casts the input to a string value. If the input is a boolean, it will be cast to `'true'` if it is `true` and `'false'` if it is `false`. If the input is `NULL`, the output will be `NULL`.

_Example:_

We are given `table1` with columns `a` and `b` and `c`

```python
table1 = pd.DataFrame({
    'a': [1.1, 0, 2],
    'b': [True, False, True],
    'c': [None, 1, 0]
})
```

upon query

```sql
SELECT
    TO_CHAR(a) AS a,
    TO_CHAR(b) AS b,
    TO_CHAR(c) AS c
FROM table1;
```

we will get the following output:

```
    a      b      c
0  1.1   true   <NA>
1    0  false      1
2    2   true      0
```

Note that if the input is a `TIMESTAMP_TZ` the only currently supported output
format that includes `TZH` or `TZM` is `YYYY-MM-DD HH:MM:SS.SSSSSSSSS +TZH:TZM`
(where `+` represents `+` or `-`). Formats that do not include those
identifiers are supported.
