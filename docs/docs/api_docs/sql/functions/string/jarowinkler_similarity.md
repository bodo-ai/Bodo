# JAROWINKLER_SIMILARITY


`#!sql JAROWINKLER_SIMILARITY(string0, string1)`

Computes the Jaro-Winkler similarity between `#!sql string0`
and `#!sql string1` as an integer between 0 and 100 (with 0
being no similarity and 100 being an exact match). The computation
is not case-sensitive, but is sensitive to spaces or formatting
characters. A scaling factor of 0.1 is used for the computation.
For the definition of Jaro-Winkler similarity, [see here](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance).



