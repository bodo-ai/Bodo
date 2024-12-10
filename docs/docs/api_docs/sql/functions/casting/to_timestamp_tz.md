# TO_TIMESTAMP_TZ

`#!sql TO_TIMESTAMP_TZ(EXPR)`

Equivalent to `#!sql TO_TIMESTAMP` except that if the input is a timezone-aware
timestamp, then the timezone's offset for at the value specified by the
timestamp is used as the `TIMESTAMPTZ` UTC offset, otherwise the local time
zone's UTC offset at the value specified by `TO_TIMESTAMP(EXPR)` is used. For
example, in the `America/Los Angeles` timezone, then the following would be
true:

```sql
TO_TIMESTAMP_TZ('2024-03-10 00:00:00'::timestampltz) = '2024-03-10 00:00:00 -0800'::timestamptz

TO_TIMESTAMP_TZ('2024-03-11 00:00:00'::timestampltz) = '2024-03-11 00:00:00 -0700'::timestamptz
```

Additionally, if `EXPR` evaluates to a string, if an offset is not explicitly
specified, the offset of the timestamp in the session's timezone is used. The
following formats for offset are supported:

- `z` or `Z` for the zero offset
- `[+-]H:M`
- `[+-]HH:M`
- `[+-]H:MM`
- `[+-]HH:MM`
- `[+-]HHMM`
- `[+-]HMM`
- `[+-]HH`
