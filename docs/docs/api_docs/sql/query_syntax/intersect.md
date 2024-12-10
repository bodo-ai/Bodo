# INTERSECT

The `#!sql INTERSECT` operator is used to calculate the intersection of
two `#!sql SELECT` statements:

```sql
SELECT <COLUMN_NAMES> FROM <TABLE1>
INTERSECT
SELECT <COLUMN_NAMES> FROM <TABLE2>
```

Each `#!sql SELECT` statement within the `#!sql INTERSECT` clause must have the
same number of columns. The columns must also have similar data
types. The output of the `#!sql INTERSECT` is the set of rows which are
present in both of the input SELECT statements. The `#!sql INTERSECT`
operator selects only the distinct values from the inputs.
