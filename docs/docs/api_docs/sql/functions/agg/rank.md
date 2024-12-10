# RANK

`#!sql RANK()`

Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition.
The rank begins with 1 and increments by one for each succeeding value. Duplicate value(s) will produce
the same rank, producing gaps in the rank (compare with `#!sql DENSE_RANK`). `#!sql ORDER BY` is required for this function.
