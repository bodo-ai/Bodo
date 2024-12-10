# NOT IN

The `#!sql IN` determines if a value can be chosen a list of options.
Currently, we support lists of literals or columns with matching
types:

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <COLUMN_NAME> IN (<val1>, <val2>, ... <valN>)
```

For example:

```sql
SELECT A FROM table1 WHERE A IN (5, 10, 15, 20, 25)
```

### Example Usage

```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"PAYMENTS":df})
...    query = "SELECT customerID FROM payments WHERE \"paymentType\" IN ('AMEX', 'WIRE')"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"PAYMENTS":df})
...    query = "SELECT customerID FROM payments WHERE \"paymentType\" NOT IN ('AMEX', 'VISA')"
...    res = bc.sql(query)
...    return res

>>>payment_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE", "VISA", "VISA", "WIRE", "VISA", "AMEX"],
... })

>>>g1(payment_df) # IN
   CUSTOMERID
2           2
4           4
7           7
9           9

>>>g2(payment_df) # NOT IN
   CUSTOMERID
4           4
7           7
```
