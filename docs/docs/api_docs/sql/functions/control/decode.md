# DECODE


`#!sql DECODE(Arg0, Arg1, Arg2, ...)`

When `Arg0` is `Arg1`, outputs `Arg2`. When `Arg0` is `Arg3`,
outputs `Arg4`. Repeats until it runs out of pairs of arguments.
At this point, if there is one remaining argument, this is used
as a default value. If not, then the output is `NULL`.

!!! note
    Treats `NULL` as a literal value that can be matched on.

Therefore, the following:

```sql
DECODE(A, NULL, 0, 'x', 1, 'y', 2, 'z', 3, -1)
```

Is logically equivalent to:

```sql
CASE WHEN A IS NULL THEN 0
     WHEN A = 'x' THEN 1
     WHEN A = 'y' THEN 2
     WHEN A = 'z' THEN 3
     ELSE -1 END
```

