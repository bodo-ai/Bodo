# TO_BOOLEAN


-  `#!sql TO_BOOLEAN(COLUMN_EXPRESSION)`

Casts the input to a boolean value. If the input is a string, it will be cast to `true` if it is
`'true'`, `'t'`, `'yes'`, `'y'`, `'1'`, cast to `false` if it is `'false'`, `'f'`, `'no'`, `'n'`, `'0'`,
and throw an error otherwise.
If the input is an integer, it will be cast to `true` if it is non-zero and `false` if it is zero.
If the input is a float, it will be cast to `true` if it is non-zero, `false` if it is zero, and throw an error on other inputs (e.g. `inf`) input. If the input is `NULL`, the output will be `NULL`.

_Example:_

We are given `table1` with columns `a` and `b` and `c`
```python
table1 = pd.DataFrame({
    'a': [1.1, 0, 2],
    'b': ['t', 'f', 'YES'],
    'c': [None, 1, 0]
})
```
upon query
```sql
SELECT
    TO_BOOLEAN(a) AS a,
    TO_BOOLEAN(b) AS b,
    TO_BOOLEAN(c) AS c
FROM table1;
```
we will get the following output:
```
       a      b      c
0   True   True   <NA>
1  False  False   True
2   True   True  False
```
Upon query
```sql
SELECT TO_BOOLEAN('other')
```
we see the following error:
```python
ValueError: invalid value for boolean conversion: string must be one of {'true', 't', 'yes', 'y', 'on', '1'} or {'false', 'f', 'no', 'n', 'off', '0'}
```

!!!note
BodoSQL will read `NaN`s as `NULL` and thus will not produce errors on `NaN` input.

