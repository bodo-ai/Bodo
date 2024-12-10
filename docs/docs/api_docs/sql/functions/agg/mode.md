# MODE

`#!sql MODE`

Returns the most frequent element in a column/group/window, including `#sql NULL` if that
is the element that appears the most. Supported on all non-semi-structured types.

Returns `#!sql NULL` if the input is all `#!sql NULL` or empty.

!!! note
In case of a tie, BodoSQL will choose a value arbitrarily based on performance considerations.
