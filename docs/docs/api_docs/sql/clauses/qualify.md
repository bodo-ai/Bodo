# QUALIFY



`#!sql QUALIFY` is similar to `#!sql HAVING`, except it applies filters after computing the results of at least one window function. `#!sql QUALIFY` is used after using `#!sql WHERE` and `#!sql HAVING`.

For example:

```sql
SELECT column_name(s),
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
QUALIFY MAX(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING) > 1
```

Is equivalent to

```sql
SELECT column_name(s) FROM
    (SELECT column_name(s), MAX(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING) as window_output
    FROM table_name
    WHERE condition
    GROUP BY column_name(s)
    HAVING condition)
WHERE window_output > 1
```

