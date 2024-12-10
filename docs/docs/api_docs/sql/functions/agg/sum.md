# SUM
`#!sql SUM`

Compute the sum of the column/group/window. Supported on numeric types.

Returns `#!sql NULL` if the input is all `#!sql NULL` or empty.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
