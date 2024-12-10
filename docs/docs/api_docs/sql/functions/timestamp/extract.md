# EXTRACT

`#!sql EXTRACT(TimeUnit from timestamp_val)`

Extracts the specified TimeUnit from the supplied date.

Allowed TimeUnits are:

- `MICROSECOND`
- `MINUTE`
- `HOUR`
- `DAY` (Day of Month)
- `DOY` (Day of Year)
- `DOW` (Day of week)
- `WEEK`
- `MONTH`
- `QUARTER`
- `YEAR`

TimeUnits are not case-sensitive.
