# TRY_TO_BOOLEAN

- `#!sql TRY_TO_BOOLEAN(COLUMN_EXPRESSION)`

This is similar to `#!sql TO_BOOLEAN` except that it will return `NULL` instead of throwing an error invalid inputs.

_Example:_

We are given `table1` with columns `a` and `b` and `c`

```python
table1 = pd.DataFrame({
    'a': [1.1, 0, np.inf],
    'b': ['t', 'f', 'YES'],
    'c': [None, 1, 0]
})
```

upon query

```sql
SELECT
    TRY_TO_BOOLEAN(a) AS a,
    TRY_TO_BOOLEAN(b) AS b,
    TRY_TO_BOOLEAN(c) AS c
FROM table1;
```

we will get the following output:

```
    a      b      c
0   True   True   <NA>
1  False  False   True
2   <NA>   True  False
```
