# SUBSTRING


`#!sql SUBSTRING(str, start_index, len)`

Takes a substring of the specified string, starting at the
specified index, of the specified length. `start_index = 1`
specifies the first character of the string, `start_index =
-1` specifies the last character of the string. `start_index
= 0` causes the function to return empty string. If
`start_index` is positive and greater than the length of the
string, returns an empty string. If `start_index` is
negative, and has an absolute value greater than the
length of the string, the behavior is equivalent to
`start_index = 1`.

For example:

```sql
SUBSTRING('hello world', 1, 5) =='hello'
SUBSTRING('hello world', -5, 7) =='world'
SUBSTRING('hello world', -20, 8) =='hello wo'
SUBSTRING('hello world', 0, 10) ==''
```
