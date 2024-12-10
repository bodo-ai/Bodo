# CASE

The `#!sql CASE` statement goes through conditions and returns a value
when the first condition is met:

```sql
SELECT CASE WHEN cond1 THEN value1 WHEN cond2 THEN value2 ... ELSE valueN END
```

For example:

```sql
SELECT (CASE WHEN A 1 THEN A ELSE B END) as mycol FROM table1
```

If the types of the possible return values are different, BodoSQL
will attempt to cast them all to a common type, which is currently
undefined behavior. The last else clause can optionally be
excluded, in which case, the `#!sql CASE` statement will return null if
none of the conditions are met. For example:

```sql
SELECT (CASE WHEN A < 0 THEN 0 END) as mycol FROM table1
```

is equivalent to:

```sql
SELECT (CASE WHEN A < 0 THEN 0 ELSE NULL END) as mycol FROM table1
```
