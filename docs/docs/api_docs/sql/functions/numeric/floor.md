# FLOOR

`#!sql FLOOR(X[, scale])`

Converts X to the specified scale, rounding towards negative
infinity. For example, `scale=0` down up to the nearest integer,
`scale=2` rounds down to the nearest `0.01`, and `scale=-1` rounds
down to the nearest multiple of 10.
