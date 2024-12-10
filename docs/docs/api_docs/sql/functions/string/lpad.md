# LPAD

`#!sql LPAD(string, len, padstring)`

Extends the input string to the specified length, by
appending copies of the padstring to the left of the
string. If the input string's length is less than the len
argument, it will truncate the input string.

For example:

```sql
LPAD('hello', 10, 'abc') =='abcabhello'
LPAD('hello', 1, 'abc') =='h'
```
