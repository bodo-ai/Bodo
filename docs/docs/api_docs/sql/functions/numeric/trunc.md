# TRUNC

`#!sql TRUNC(X[, num_decimal_places])`

Equivalent to `#!sql TRUNC(X[, num_decimal_places])` if `X` is numeric.
Note that `TRUNC` is overloaded and may invoke the timestamp function
`TRUNC` if `X` is a date or time expression.
