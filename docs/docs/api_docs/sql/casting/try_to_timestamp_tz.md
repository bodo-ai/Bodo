# TRY_TO_TIMESTAMP_TZ


`#!sql TRY_TO_TIMESTAMP_NTZ(EXPR)`

Equivalent to `#!sql TRY_TO_TIMESTAMP` except that it uses the local time zone, or keeps
the original timezone if the input is a timezone-aware timestamp.

