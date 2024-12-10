# LAG
`#!sql LAG(COLUMN_EXPRESSION, [N], [FILL_VALUE])`

Returns the row that precedes the current row by N. If N
is not specified, defaults to 1. If FILL_VALUE is not
specified, defaults to `NULL`. If
there are fewer than N rows the follow the current row in
the window, it returns FILL_VALUE. N must be a literal
integer if specified. FILL_VALUE must be a scalar if specified.
Supported on all non-semi-structured types.


