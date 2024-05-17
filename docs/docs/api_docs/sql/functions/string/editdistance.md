# EDITDISTANCE


`#!sql EDITDISTANCE(string0, string1[, max_distance])`

Returns the minimum edit distance between `#!sql string0` and `#!sql string1`
according to Levenshtein distance. Optionally accepts a third
argument specifying a maximum distance value. If the minimum
edit distance between the two strings exceeds this value, then
this value is returned instead. If it is negative, zero
is returned.


