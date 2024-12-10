# ARRAY_INTERSECTION

`#!sql ARRAY_INTERSECTION(A, B)`

Takes in two arrays and returns an arary of all the elements from the
first array that also appear in the second. If an element appears in
either array more than once, that element is kept the minimum of the
number of times it appears in either array. For instance, if the
first array contains three 1s and four 6s, and the second array
contains two 1s and five 6s, then the output will have two 1s and
three 6s.
