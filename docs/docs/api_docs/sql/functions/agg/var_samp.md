# VAR_SAMP
`#!sql VAR_SAMP`

Compute the variance for a column/group/window with N - 1 degrees of
freedom. Supported on numeric types.

Returns `#!sql NULL` if the input is all `#!sql NULL` or empty.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
