# SELECT DISTINCT

The `#!sql SELECT DISTINCT` statement is used to return only distinct
(different) values:

```sql
SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>
```

`#!sql DISTINCT` can be used in a SELECT statement or inside an
aggregate function. For example:

```sql
SELECT DISTINCT A FROM table1

SELECT COUNT DISTINCT A FROM table1
```

### Example Usage

```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"PAYMENTS":df})
...    query = "SELECT DISTINCT \"paymentType\" FROM payments"
...    res = bc.sql(query)
...    return res

>>>payment_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE", "VISA", "VISA", "WIRE", "VISA", "AMEX"],
... })

>>>g(payment_df) # inside SELECT
paymentType
0        VISA
2        AMEX
4        WIRE

>>>def g(df):
...    bc = bodosql.BodoSQLContext({"PAYMENTS":df})
...    query = "SELECT COUNT(DISTINCT \"paymentType\") as num_payment_types FROM payments"
...    res = bc.sql(query)
...    return res

>>>g(payment_df) # inside aggregate
NUM_PAYMENT_TYPES
0          3
```
