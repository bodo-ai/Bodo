# GROUP BY

The `#!sql GROUP BY` statement groups rows that have the same values
into summary rows, like "find the number of customers in each
country". The `#!sql GROUP BY` statement is often used with aggregate
functions to group the result-set by one or more columns:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <CONDITION>
GROUP BY <GROUP_EXPRESSION>
ORDER BY <COLUMN_NAMES>
```

For example:
```sql
SELECT MAX(A) FROM table1 GROUP BY B
```
`#!sql GROUP BY` statements also referring to columns by alias or
column number:
```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val
SELECT MAX(A), B FROM table1 GROUP BY 2
```

BodoSQL supports several subclauses that enable grouping by multiple different
sets of columns in the same `#!sql SELECT` statement. `#!sql GROUPING SETS` is the first. It is
equivalent to performing a group by for each specified set (setting each column not
present in the grouping set to null), and unioning the results. For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS (B, B, (B, C), ())
```

This is equivalent to:

```sql
SELECT * FROM
    (SELECT MAX(A), B, null FROM table1 GROUP BY B)
UNION
    (SELECT MAX(A), B, null FROM table1 GROUP BY B)
UNION
    (SELECT MAX(A), B, C FROM table1 GROUP BY B, C)
UNION
    (SELECT MAX(A), null, null FROM table1)
```

!!! note
    The above example is not valid BodoSQL code, as we do not support null literals.
    It is used only to show the null filling behavior.

`#!sql CUBE` is equivalent to grouping by all possible permutations of the specified set.
For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY CUBE(B, C)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C), (B), (C), ())
```

`#!sql ROLLUP` is equivalent to grouping by n + 1 grouping sets, where each set is constructed by dropping the rightmost element from the previous set, until no elements remain in the grouping set. For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY ROLLUP(B, C, D)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), ())
```

`#!sql CUBE` and `#!sql ROLLUP` can be nested into a `#!sql GROUPING SETS` clause. For example:

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS (ROLLUP(B, C, D), CUBE(B, C), (A))
```

Which is equivalent to

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), (), (B, C), (B), (C), (), (A))
```

