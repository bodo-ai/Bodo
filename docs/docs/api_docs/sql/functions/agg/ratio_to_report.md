# RATIO_TO_REPORT

`#!sql RATIO_TO_REPORT(COLUMN_EXPRESSION)`

Returns an element in the window frame divided by the sum of all elements in the
same window frame, or `NULL` if the window frame has a sum of zero. For example,
if calculating `#!sql RATIO_TO_REPORT` on `[2, 5, NULL, 10, 3]` where the window
is the entire partition, the answer is `[0.1, 0.25, NULL, 0.5, 0.15]`.
