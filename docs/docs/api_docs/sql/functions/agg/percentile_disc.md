# PERCENTILE_DISC
`#!sql PERCENTILE_DISC(q) WITHIN GROUP (ORDER BY A)`

Computes the exact value of the `q`-th percentile of column `A` (e.g.
0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
and `q` can be any scalar float between zero and one.

This function differs from `PERCENTILE_CONT` in that it always outputs a
value from the original array. The value it chooses is the smallest value
in `A` such that the `CUME_DIST` of all values in the column `A` is greater
than or equal to `q`. For example, consider the dataset `[2, 8, 8, 40]`.
The `CUME_DIST` of each of these values is `[0.25, 0.75, 0.75, 1.0]`.
If we sought the percentile `q=0.6` we would output 8 since it has the
smallest `CUME_DIST` that is `>=0.6`.


