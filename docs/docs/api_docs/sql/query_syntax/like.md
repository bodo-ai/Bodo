# LIKE



The `#!sql LIKE` clause is used to filter the strings in a column to
those that match a pattern:
```sql
SELECT column_name(s) FROM table_name WHERE column LIKE pattern
```
In the pattern we support the wildcards `#!sql %` and `#!sql _`. For example:
```sql
SELECT A FROM table1 WHERE B LIKE '%py'
```

