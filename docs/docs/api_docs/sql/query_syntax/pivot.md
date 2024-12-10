# PIVOT

The `#!sql PIVOT` clause is used to transpose specific data rows in one
or more columns into a set of columns in a new DataFrame:

```sql
SELECT col1, ..., colN FROM table_name PIVOT (
    AGG_FUNC_1(colName or pivotVar) AS alias1, ...,  AGG_FUNC_N(colName or pivotVar) as aliasN
    FOR pivotVar IN (ROW_VALUE_1 as row_alias_1, ..., ROW_VALUE_N as row_alias_N)
)
```

`#!sql PIVOT` produces a new column for each pair of pivotVar and
aggregation functions.

For example:

```sql
SELECT single_sum_a, single_avg_c, triple_sum_a, triple_avg_c FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR A IN (1 as single, 3 as triple)
)
```

Here `#!sql single_sum_a` will contain sum(A) where `#!sql A = 1`,
single_avg_c will contain AVG(C) where `#!sql A = 1` etc.

If you explicitly specify other columns as the output, those
columns will be used to group the pivot columns. For example:

```sql
SELECT B, single_sum_a, single_avg_c, triple_sum_a, triple_avg_c FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR A IN (1 as single, 3 as triple)
)
```

Contains 1 row for each unique group in B. The pivotVar can also
require values to match in multiple columns. For example:

```sql
SELECT * FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR (A, B) IN ((1, 4) as col1, (2, 5) as col2)
)
```
