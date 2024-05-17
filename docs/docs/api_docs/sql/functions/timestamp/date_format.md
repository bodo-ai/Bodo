# DATE_FORMAT


`#!sql DATE_FORMAT(timestamp_val, literal_format_string)`

Converts a timestamp value to a String value given a
scalar format string.

Recognized formatting characters:

`#!sql %i` Minutes, zero padded (00 to 59)
`#!sql %M` Full month name (January to December)
`#!sql %r` Time in format in the format (hh\:mm\:ss AM/PM)
`#!sql %s` Seconds, zero padded (00 to 59)
`#!sql %T` Time in format in the format (hh\:mm\:ss)
`#!sql %T` Time in format in the format (hh\:mm\:ss)
`#!sql %u` week of year, where monday is the first day of the week(00 to 53)
`#!sql %a` Abbreviated weekday name (sun-sat)
`#!sql %b` Abbreviated month name (jan-dec)
`#!sql %f` Microseconds, left padded with 0's, (000000 to 999999)
`#!sql %H` Hour, zero padded (00 to 23)
`#!sql %j` Day Of Year, left padded with 0's (001 to 366)
`#!sql %m` Month number (00 to 12)
`#!sql %p` AM or PM, depending on the time of day
`#!sql %d` Day of month, zero padded (01 to 31)
`#!sql %Y` Year as a 4 digit value
`#!sql %y` Year as a 2 digit value, zero padded (00 to 99)
`#!sql %U` Week of year, where Sunday is the first day of the week
    (00 to 53)
`#!sql %S` Seconds, zero padded (00 to 59)

For example:

```sql
DATE_FORMAT(Timestamp '2020-01-12', '%Y %m %d') =='2020 01 12'
DATE_FORMAT(Timestamp '2020-01-12 13:39:12', 'The time was %T %p. It was a %u') =='The time was 13:39:12 PM. It was a Sunday'
```


