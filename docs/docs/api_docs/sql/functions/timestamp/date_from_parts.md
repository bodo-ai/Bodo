# DATE_FROM_PARTS

- `DATE_FROM_PARTS(year, month, day)`

Constructs a date from the integer inputs specified, e.g. `(2020, 7, 4)`
will output July 4th, 2020.

!!! note
Month does not have to be in the 1-12 range, and day does not have to
be in the 1-31 range. Values out of bounds are overflowed logically,
e.g. `(2020, 14, -1)` will output January 31st, 2021.
