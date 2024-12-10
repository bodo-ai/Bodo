# ARRAY_EXCEPT

`#!sql ARRAY_EXCEPT(A, B)`

Takes in two arrays and returns a copy of the first array but with all
of the elements from the second array dropped. If an element appears in
the first array more than once, that element is only dropped as many
times as it appears in the second array. For instance, if the
first array contains three 1s and four 6s, and the second array
contains two 1s and one 6, then the output will have one 1 and
three 6s.
