# UNIFORM

`#!sql UNIFORM(lo, hi, gen)`

Outputs a random number uniformly distributed in the interval `[lo, hi]`.
If `lo` and `hi` are both integers, then the output is an integer between
`lo` and `hi` (including both endpoints). If either `lo` or `hi` is a float,
the output is a random float between them. The values of `gen` are used to
seed the randomness, so if `gen` is all distinct values (or is randomly
generated) then the output of `UNIFORM` should be random. However, if 2
rows have the same `gen` value they will produce the same output value.
