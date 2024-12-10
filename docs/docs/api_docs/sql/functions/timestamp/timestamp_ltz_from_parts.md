# TIMESTAMP_LTZ_FROM_PARTS


-   `TIMESTAMP_LTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`

Equivalent to `TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`
but without the optional timezone argument in the first overload. The output
is always timezone-aware using the local timezone.


