# TIMESTAMPDIFF

`#!sql TIMESTAMPDIFF(unit, timestamp_val1, timestamp_val2)`

Returns the amount of time that has passed since `timestamp_val1` until
`timestamp_val2` in terms of the unit specified, ignoring all smaller units.
E.g., December 31 of 2020 and January 1 of 2021 count as 1 year apart.

!!! note
For all units larger than `#!sql NANOSECOND`, the output type is `#!sql INTEGER`
instead of `#!sql BIGINT`, so any difference values that cannot be stored as
signed 32-bit integers might not be returned correct.
