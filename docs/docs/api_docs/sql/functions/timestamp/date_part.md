# DATE_PART


`#!sql DATE_PART(unit, timestamp_val)`

Equivalent to `#!sql EXTRACT(unit FROM timestamp_val)` with the following unit
string literals:

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
-   TZH
-   TZM

Supported with timezone-aware data. Note that `TZH`/`TZM` are only supported for
`TIMESTAMP_TZ` inputs and extracts the offset hours and offset minutes
respectively.

