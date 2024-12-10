# APPROX_PERCENTILE

`#!sql APPROX_PERCENTILE(A, q)`

Returns the approximate value of the `q`-th percentile of column `A` (e.g.
0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
and `q` can be any scalar float between zero and one.

The approximation is calculated using the t-digest algorithm.
