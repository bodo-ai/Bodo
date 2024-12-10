# Aliasing

In all but the most trivial cases, BodoSQL generates internal names to
avoid conflicts in the intermediate Dataframes. By default, BodoSQL
does not rename the columns for the final output of a query using a
consistent approach. For example the query:

```sql
bc.sql("SELECT SUM(A) FROM table1 WHERE B > 4")
```
Results in an output column named `$EXPR0`. To reliably reference this
column later in your code, we highly recommend using aliases for all
columns that are the final outputs of a query, such as:

```py
bc.sql("SELECT SUM(A) as sum_col FROM table1 WHERE B > 4")
```

!!! note
     BodoSQL supports using aliases generated in `#!sql SELECT` inside
    `#!sql GROUP BY` and `#!sql HAVING` in the same query, but you cannot do so with
    `#!sql WHERE`
