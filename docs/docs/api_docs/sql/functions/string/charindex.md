# CHARINDEX

`#!sql CHARINDEX(str1, str2[, start_position])`

Equivalent to `#!sql POSITION(str1, str2)` when 2 arguments are provided. When the
optional third argument is provided, it only starts searching at that index.

!!! note
Not currently supported on binary data.
