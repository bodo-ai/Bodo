# RPAD

`#!sql RPAD(string, len, padstring)`

Extends the input string to the specified length, by
appending copies of the padstring to the right of the
string. If the input string's length is less than the len
argument, it will truncate the input string.

For example:

```sql
RPAD('hello', 10, 'abc') =='helloabcab'
RPAD('hello', 1, 'abc') =='h'
```
