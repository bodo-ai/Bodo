# DATEADD


`#!sql DATEADD(unit, amount, timestamp_val)`

Computes a timestamp column by adding the amount of the specified unit
to the timestamp val. For example, `#!sql DATEADD('day', 3, T)` adds 3 days to
column `T`. Allows the following units, with the specified
abbreviations as string literals:

-   YEAR: `year`, `years`, `yr`, `yrs`, `y`, `yy`, `yyy`, `yyyy`
-   QUARTER: `quarter`, `quarters`, `q`, `qtr`, `qtrs`
-   MONTH: `month`, `months`, `mm`, `mon`, `mons`
-   WEEK: `week`, `weeks`, `weekofyear`, `w`, `wk`, `woy`, `wy`
-   DAY: `day`, `days`, `dayofmonth`, `d`, `dd`
-   HOUR: `hour`, `hours`, `hrs`, `h`, `hr`, `hrs`
-   MINUTE: `minute`, `minutes`, `m`, `mi`, `min`, `mins`
-   SECOND: `second`, `seconds`, `s`, `sec`, `secs`
-   MILLISECOND: `millisecond`, `milliseconds`, `ms`, `msecs`
-   MICROSECOND: `microsecond`, `microseconds`, `us`, `usec`
-   NANOSECOND: `nanosecond`, `nanoseconds`, `nanosec`, `nsec`, `nsecs`, `nsecond`, `ns`, `nanonsecs`

Supported with timezone-aware data.

`#!sql DATEADD(timestamp_val, amount)`

Equivalent to `#!sql DATEADD('day', amount, timestamp_val)`


