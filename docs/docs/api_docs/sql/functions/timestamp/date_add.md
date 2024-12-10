# DATE_ADD

`#!sql DATE_ADD(timestamp_val, interval)`

Computes a timestamp column by adding an interval column/scalar to a
timestamp value. If the first argument is a string representation of a
timestamp, Bodo will cast the value to a timestamp.

`#!sql DATE_ADD(timestamp_val, amount)`

Equivalent to `#!sql DATE_ADD('day', amount, timestamp_val)`
