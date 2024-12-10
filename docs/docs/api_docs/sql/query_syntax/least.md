# LEAST



The `#!sql LEAST` clause is used to return the smallest value from a
list of columns:
```sql
SELECT LEAST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT LEAST(A, B, C) FROM table1
```

