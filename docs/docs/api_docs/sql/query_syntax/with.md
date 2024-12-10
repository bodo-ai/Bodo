# WITH

The `#!sql WITH` clause can be used to name sub-queries:

```sql
WITH sub_table AS (SELECT column_name(s) FROM table_name)
SELECT column_name(s) FROM sub_table
```

For example:

```sql
WITH subtable as (SELECT MAX(A) as max_al FROM table1 GROUP BY B)
SELECT MAX(max_val) FROM subtable
```
