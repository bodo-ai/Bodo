# TIME_FROM_PARTS

`#!sql TIME_FROM_PARTS(integer_hour_val, integer_minute_val, integer_second_val [, integer_nanoseconds_val])`

Creates a time from individual numeric components. Usually,
`integer_hour_val` is in the 0-23 range, `integer_minute_val` is in the 0-59
range, `integer_second_val` is in the 0-59 range, and
`integer_nanoseconds_val` (if provided) is a 9-digit integer.

```sql
TIMEFROMPARTS(12, 34, 56, 987654321)
12:34:56.987654321
```
