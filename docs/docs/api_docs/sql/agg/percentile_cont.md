# APPROX_PEPERCENTILE_CONTRCENTILE
`#!sql APPROX_PEPERCENTILE_CONTRCENTILE(q) WITHIN GROUP (ORDER BY A)`

Computes the exact value of the `q`-th percentile of column `A` (e.g.
0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
and `q` can be any scalar float between zero and one.

If no value lies exactly at the desired percentile, the two nearest
values are linearly interpolated. For example, consider the dataset `[2, 8, 25, 40]`.
If we sought the percentile `q=0.25` we would be looking for the value
at index 0.75. There is no value at index 0.75, so we linearly interpolate
between 2 and 8 to get 6.5.


