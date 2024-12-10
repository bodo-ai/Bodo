# HAVING

The `#!sql HAVING` clause is used for filtering with `#!sql GROUP BY`.
`#!sql HAVING` applies the filter after generating the groups, whereas
`#!sql WHERE` applies the filter before generating any groups:

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
```

For example:

```sql
SELECT MAX(A) FROM table1 GROUP BY B HAVING C < 0
```

`#!sql HAVING` statements also referring to columns by aliases used in
the `#!sql GROUP BY`:

```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val HAVING val 5
```
