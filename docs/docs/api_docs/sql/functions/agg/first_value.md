# FIRST_VALUE
`#!sql FIRST_VALUE(COLUMN_EXPRESSION)`

Select the first value in the window or `NULL` if the window
is empty. Supported on all non-semi-structured types.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
