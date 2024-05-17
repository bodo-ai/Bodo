# TIMESTAMP_TZ_FROM_PARTS


-   `TIMESTAMP_TZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`

Returns a TIMESTAMP_TZ constructed with the specified date/time components using the offset of the provided
timezone at that time of year. If no timezone is provided, the session timezone is used.
