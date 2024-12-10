# CONV


`#!sql CONV(X, current_base, new_base)`

`#!sql CONV` takes a string representation of an integer value,
it's current_base, and the base to convert that argument
to. `#!sql CONV` returns a new string, that represents the value in
the new base. `#!sql CONV` is only supported for converting to/from
base 2, 8, 10, and 16.

For example:

```sql
CONV('10', 10, 2) =='1010'
CONV('10', 2, 10) =='2'
CONV('FA', 16, 10) =='250'
```
