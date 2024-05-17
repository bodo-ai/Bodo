# PERCENT_RANK
`#!sql PERCENT_RANK()`

Compute the percentage ranking of the value(s) in each row based on the value(s) relative to all value(s)
within the window partition. Ranking calculated using `#!sql RANK()` divided by the number of rows in the window
partition minus one. Partitions with one row have `#!sql PERCENT_RANK()` of 0. `#!sql ORDER BY` is required for this function.


