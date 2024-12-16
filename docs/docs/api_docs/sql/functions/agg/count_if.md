# COUNT_IF
`#!sql COUNT_IF`

Compute the total number of occurrences of `#!sql true` in a column/group/window
of booleans. For example:

```sql
SELECT COUNT_IF(A) FROM table1
```

Is equivalent to
```sql
SELECT SUM(CASE WHEN A THEN 1 ELSE 0 END) FROM table1
```

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
