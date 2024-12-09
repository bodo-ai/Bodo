# AVG
`#!sql AVG`

Compute the mean of the the column/group/window. Supported
on all numeric types.


!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
