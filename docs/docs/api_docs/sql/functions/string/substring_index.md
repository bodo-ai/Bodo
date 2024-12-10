# SUBSTRING_INDEX

`#!sql SUBSTRING_INDEX(str, delimiter_str, n)`

Returns a substring of the input string, which contains
all characters that occur before n occurrences of the
delimiter string. if n is negative, it will return all
characters that occur after the last n occurrences of the
delimiter string. If `num_occurrences` is 0, it will return
the empty string

For example:

```sql
SUBSTRING_INDEX('1,2,3,4,5', ',', 2) =='1,2'
SUBSTRING_INDEX('1,2,3,4,5', ',', -2) =='4,5'
SUBSTRING_INDEX('1,2,3,4,5', ',', 0) ==''
```
