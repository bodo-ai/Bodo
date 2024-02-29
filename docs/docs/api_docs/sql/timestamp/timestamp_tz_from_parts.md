# TIMESTAMP_TZ_FROM_PARTS


-   `TIMESTAMP_TZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`

Equivalent to `TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`
except the default behavior if no timezone is provided is to use the local
timezone instead of timezone-naive.

!!! note
    Timezone argument is not supported at this time.


