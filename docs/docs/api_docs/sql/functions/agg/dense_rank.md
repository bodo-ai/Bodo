# DENSE_RANK
`#!sql DENSE_RANK()`

Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition
without producing gaps in the rank (compare with `#!sql RANK`). The rank begins with 1 and increments by one for each succeeding value.
Rows with the same value(s) produce the same rank. `#!sql ORDER BY` is required for this function.

!!!note
    To compare `#!sql RANK` and `#!sql DENSE_RANK`, on input array `['a', 'b', 'b', 'c']`, `#!sql RANK` will output `[1, 2, 2, 4]` while `#!sql DENSE_RANK` outputs `[1, 2, 2, 3]`.


