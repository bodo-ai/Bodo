# DATEDIFF


`#!sql DATEDIFF(timestamp_val1, timestamp_val2)`

Computes the difference in days between two Timestamp
values (timestamp_val1 - timestamp_val2)


`#!sql DATEDIFF(unit, timestamp_val1, timestamp_val2)`

Computes the difference between two Timestamp
values (timestamp_val2 - timestamp_val1) in terms of unit

Allows the following units, with the specified
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

