# TO_BINARY

- `TO_BINARY(COLUMN_EXPRESSION)`

Casts the input string to binary data. Currently only supports the `HEX` format.
Raises an exception if the input is not a valid hex string:

- Must have an even number of characters
- All characters must be hexedecimal digits (0-9, a-f case insensitive)

_Example:_

We are given `table1` with columns `a` and `b`:

```python
table1 = pd.DataFrame({
    'a': ["AB", "626f646f", "4a2F3132"],
    'b': ["ABC", "ZETA", "#fizz"],
})
```

upon query

```sql
SELECT TO_BINARY(a),
FROM table1
```

we will get the following output:

```
    TO_BINARY(a)
0   b'\xab'             -- Binary encoding of the character '¼'
1   b'\x62\x6f\x64\x6f' -- Binary encoding of the string 'bodo'
2   b'\x4a\x2f\x31\x32' -- Binary encoding of the string 'J/12'
```

Upon query

```sql
SELECT TO_BINARY(b)
FROM table1
```

we will see a value error because all of the values in column b are not valid
hex strings:

- `'ABC'` is 3 characters, which is not an even number
- `'ZETA'` contains non-hex characters `Z` and `T`
- `'#fizz'` is 5 characters, which is not an even number and contains non-hex
  characters `#`, `i` and `z`
