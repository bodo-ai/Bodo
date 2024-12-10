# TO_DOUBLE

- `#!sql TO_DOUBLE(COLUMN_EXPRESSION)`

Converts a numeric or string expression to a double-precision floating-point number.
For `NULL` input, the result is `NULL`.
Fixed-point numbers are converted to floating point; the conversion cannot
fail, but might result in loss of precision.
Strings are converted as decimal or integer numbers. Scientific notation
and special values (nan, inf, infinity) are accepted, case insensitive.

_Example:_

We are given `table1` with columns `a` and `b`

```python
table1 = pd.DataFrame({
    'a': [1, 0, 2],
    'b': ['3.7', '-2.2e-1', 'nan'],
})
```

upon query

```sql
SELECT
    TO_DOUBLE(a) AS a,
    TO_DOUBLE(b) AS b,
FROM table1;
```

we will get the following output:

```
       a      b
0    1.0    3.7
1    0.0  -0.22
2    2.0   <NA>
```
