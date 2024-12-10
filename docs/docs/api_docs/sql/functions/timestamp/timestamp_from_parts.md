# TIMESTAMP_FROM_PARTS

`#!sql TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`
`#!sql TIMESTAMP_FROM_PARTS(date_expr, time_expr)`
The first overload is equivalent to `DATE_FROM_PARTS` but also takes in an
hour, minute and second (which can be out of bounds just like the
month/day). Optionally takes in a nanosecond value, and a timezone value
for the output. If the timezone is not specified, the output is
timezone-naive. Note that if any numeric argument cannot be converted to
an int64, then it will become NULL.

!!! note
Timezone argument is not supported at this time.

The second overload constructs the timestamp by combining the date and time
arguments. The output of this function is always timestamp-naive.
