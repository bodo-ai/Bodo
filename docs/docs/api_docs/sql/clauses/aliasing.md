# Aliasing



SQL aliases are used to give a table, or a column in a table, a
temporary name:

```sql
SELECT <COLUMN_NAME> AS <ALIAS>
FROM <TABLE_NAME>
```

For example:
```sql
Select SUM(A) as total FROM table1
```

We strongly recommend using aliases for the final outputs of any
queries to ensure all column names are predictable.
