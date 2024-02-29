# GREATEST



The `#!sql GREATEST` clause is used to return the largest value from a
list of columns:
```sql
SELECT GREATEST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT GREATEST(A, B, C) FROM table1
```

