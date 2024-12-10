# STRTOK

`#!sql STRTOK(source[, delimiter[, part]])`

Tokenizes the source string by occurrences of any character in the
delimiter string and returns the occurrence specified by the part.
I.e. if part=1, returns the substring before the first occurrence,
and if part=2, returns the substring between the first and second
occurrence. Zero and negative indices are not allowed. Empty tokens
are always skipped in favor of the next non-empty token. In any
case where the only possible output is '', the output is `NULL`.
The delimiter is optional and defaults to ' '. The part is optional
and defaults to 1.
