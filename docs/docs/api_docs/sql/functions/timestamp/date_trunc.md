# DATE_TRUNC

`#!sql DATE_TRUNC(str_literal, timestamp_val)`

Truncates a timestamp to the provided str_literal field.
str_literal must be a compile time constant and one of:

- "MONTH"
- "WEEK"
- "DAY"
- "HOUR"
- "MINUTE"
- "SECOND"
- "MILLISECOND"
- "MICROSECOND"
- "NANOSECOND"
