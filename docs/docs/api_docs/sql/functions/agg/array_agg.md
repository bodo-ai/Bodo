# ARRAY_AGG

`#!sql ARRAY_AGG([DISTINCT] A) [WITHIN GROUP(ORDER BY orderby_terms)]`

Combines all the values in column `A` within each group into a single array.

Optionally allows using a `WITHIN GROUP` clause to specify how the values should
be ordered before being combined into an array. If no clause is specified, then the ordering
is unpredictable. Nulls will not be included in the arrays.

If the `DISTINCT` keyword is provided, then duplicate elements are removed from each of
the arrays. However, if this keyword is provied and a `WITHIN GROUP` clause is also provided,
then the `WITHIN GROUP` clause can only refer to the same column as the aggregation input.
