# TO_TIMESTAMP_LTZ

`#!sql TO_TIMESTAMP_LTZ(EXPR)`

Equivalent to `#!sql TO_TIMESTAMP` except that it uses the local time zone.
If `EXPR` evaluates to a `TIMESTAMP_TZ`, then the offset will be removed (local
with respect to the offset), and the timezone will be set the local time zone.
Note that the output will be converted to a `TIMESTAMP_LTZ` (output will have a
defined timezone instead of a constant offset).
