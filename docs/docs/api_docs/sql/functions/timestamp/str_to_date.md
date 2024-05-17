# STR_TO_DATE


`#!sql STR_TO_DATE(str_val, literal_format_string)`

Converts a string value to a Timestamp value given a
literal format string. If a year, month, and day value is
not specified, they default to 1900, 01, and 01
respectively. Will throw a runtime error if the string
cannot be parsed into the expected values. See [`DATE_FORMAT`][date_format]
for recognized formatting characters.

For example:

```sql
STR_TO_DATE('2020 01 12', '%Y %m %d') ==Timestamp '2020-01-12'
STR_TO_DATE('01 12', '%m %d') ==Timestamp '1900-01-12'
STR_TO_DATE('hello world', '%Y %m %d') ==RUNTIME ERROR
```

