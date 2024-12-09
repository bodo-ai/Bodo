# NTH_VALUE
`#!sql NTH_VALUE(COLUMN_EXPRESSION, N)`

Select the last value in the window or `NULL` if the window
does not have `N` elements. Uses 1-indexing. Requires
`N` to be a positive integer literal.  Supported on all
non-semi-structured types.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
