# SPLIT_PART

`#!sql SPLIT_PART(source, delimiter, part)`

Returns the substring of the source between certain occurrence of
the delimiter string, the occurrence being specified by the part.
I.e. if part=1, returns the substring before the first occurrence,
and if part=2, returns the substring between the first and second
occurrence. Zero is treated like 1. Negative indices are allowed.
If the delimiter is empty, the source is treated like a single token.
If the part is out of bounds, '' is returned.
