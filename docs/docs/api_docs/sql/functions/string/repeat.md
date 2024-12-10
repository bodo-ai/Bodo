# REPEAT

`#!sql REPEAT(str, len)`

Extends the specified string to the specified length by
repeating the string. Will truncate the string If the
string's length is less than the len argument

For example:

```sql
REPEAT('abc', 7) =='abcabca'
REPEAT('hello world', 5) =='hello'
```
