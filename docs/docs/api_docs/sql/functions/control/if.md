# IF

`#!sql IF(Cond, TrueValue, FalseValue)`

Returns `TrueValue` if cond is true, and `FalseValue` if cond is
false. Logically equivalent to:

```sql
CASE WHEN Cond THEN TrueValue ELSE FalseValue END
```
