# TIMESTAMP_NTZ_FROM_PARTS


-   `TIMESTAMP_NTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`
-   `TIMESTAMP_NTZ_FROM_PARTS(date_expr, time_expr)`

Equivalent to `TIMESTAMP_FROM_PARTS` but without the optional timezone
argument in the first overload. The output is always timezone-naive.


