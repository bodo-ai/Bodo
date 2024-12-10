# TRUNC


`#!sql TRUNC(timestamp_val, str_literal)`

Equivalent to `#!sql DATE_TRUNC(str_literal, timestamp_val)`. The
argument order is reversed when compared to `DATE_TRUNC`. Note that `TRUNC`
is overloaded, and may invoke the numeric function `TRUNCATE` if the
arguments are numeric.

