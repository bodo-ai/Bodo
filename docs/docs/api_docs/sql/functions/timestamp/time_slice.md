# TIME_SLICE


`#!sql TIME_SLICE(date_or_time_expr, slice_length, unit[, start_or_end])`

Calculates one of the endpoints of a "slice" of time containing the date
specified by `date_or_time_expr` where each slice has length of time corresponding
to `slice_length` times the date/time unit specified by `unit`. The slice
start/ends are always aligned to the unix epoch `1970-01-1` (at midnight). The fourth argument
specifies whether to return the begining or the end of the slice
(`'START'` for begining, `'END'` for end), where the default is `'START'`.

For example, `#!sql TIME_SLICE(T, 3, 'YEAR')` would return the timestamp
corresponding to the begining of the first 3-year window (aligned with
1970) that contains timestamp `T`. So `T = 1995-7-4 12:30:00` would
output `1994-1-1` for `'START'` or `1997-1-1` for `'END'`.


