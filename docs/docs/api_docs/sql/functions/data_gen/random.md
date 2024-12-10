# RANDOM


`#!sql RANDOM()`

Outputs a random 64-bit integer. If used inside of a select statement with
a table, the number of random values will match the number of rows in the
input table (and each value should be randomly and independently generated).
Note that running with multiple processors may affect the randomization
results.

!!! note
    Currently, BodoSQL does not support the format of `#!sql RANDOM()` that
    takes in a seed value.

!!! note
    At present, aliases to `RANDOM` calls occasionally produce unexpected
    behavior. For certain SQL operations, calling `RANDOM` and storing the
    result with an alias, then later re-using that alias may result in
    another call to `RANDOM`. This behavior is somewhat rare.


