# Supported DataFrame Data Types

BodoSQL uses its internal Python tables to represent SQL tables in memory and
converts SQL types to corresponding Python types which are used by Bodo.
Below is a table mapping SQL types used in BodoSQL to their respective
Python types and Bodo data types.


<center>

| SQL Type(s)           | Equivalent Python Type | Bodo Data Type                       |
|-----------------------|------------------------|--------------------------------------|
| `BOOLEAN`             | `np.bool_`             | `bodo.types.bool_`                         |
| `TINYINT`             | `np.int8`              | `bodo.types.int8`                          |
| `SMALLINT`            | `np.int16`             | `bodo.types.int16`                         |
| `INT`                 | `np.int32`             | `bodo.types.int32`                         |
| `BIGINT`              | `np.int64`             | `bodo.types.int64`                         |
| `FLOAT`               | `np.float32`           | `bodo.types.float32`                       |
| `DOUBLE`              | `np.float64`           | `bodo.types.float64`                       |
| `VARCHAR`, `CHAR`     | `str`                  | `bodo.types.string_type`                   |
| `VARBINARY`, `BINARY` | `bytes`                | `bodo.types.bytes_type`                    |
| `DATE`                | `datetime.date`        | `bodo.types.datetime_date_type`            |
| `TIME`                | `bodo.types.Time`            | `bodo.types.TimeType`                      |
| `TIMESTAMP_NTZ`       | `pd.Timestamp`         | `bodo.types.PandasTimestampType(None)`     |
| `TIMESTAMP_LTZ`       | `pd.Timestamp`         | `bodo.types.PandasTimestampType(local_tz)` |
| `TIMESTAMP_TZ`        | `bodo.types.TimestampTZ`     | `bodo.types.timestamptz_type`              |
| `INTERVAL(day-time)`  | `np.timedelta64[ns]`   | `bodo.types.timedelta64ns`                 |
| `ARRAY`               | `pyarrow.large_list`   | `bodo.ArrayItemArray`                |
| `MAP`                 | `pyarrow.map`          | `bodo.MapScalarType`                 |
| `NULL`                | `pyarrow.NA`           | `bodo.types.null_dtype`                    |

</center>

BodoSQL may be able to handle additional column types if the data is unused. When loading
data from Snowflake or other sources, BodoSQL will treat Decimal columns as either BigInt
or Float64 depending on the column's scale and precision.

### Unsigned Types

Although SQL does not explicitly support unsigned types,
BodoSQL typically maintains the types of the existing DataFrames registered
in a [BodoSQLContext]. If these types are unsigned, then this may result in
different behavior than expected. We always recommend working with signed types
to avoid any potential issues.

### TIMESTAMP\_TZ

Note that `bodo.types.TimestampTZ` in python is a custom type provided by the Bodo
library. In `sql` this datatype is compatible with [Snowflake's
TIMESTAMP\_TZ](https://docs.snowflake.com/en/sql-reference/data-types-datetime#timestamp-ltz-timestamp-ntz-timestamp-tz).

`TIMESTAMP_TZ` stores a timestamp along with a `UTC` offset with a resolution of
minutes. This offset can be arbitrary, but it is not dependant on the timestamp
value. In other words, it is not aware of timezones and changes in offset such
as `DST`. While most operations will use the timestamp value (not `UTC`), any
comparison between two `TIMESTAMP_TZ` values will treat them as equal if
their `UTC` time is equal. For example:

```sql
SELECT '2024-01-01 00:00:00 +00:00'::timestamptz = '2024-01-01 01:00:00 +01:00'::timestamptz
```
The above query will output a row with `TRUE`  - the timestamps *are* the same
with respect to `UTC` even though their values without the offset are different.

```sql
SELECT '2024-01-01 00:00:00 +00:00'::timestamptz = '2024-01-01 00:00:00 +05:00'::timestamptz
```
The above query will output a row with `False` - the timestamps *are not* the same
with respect to `UTC` even though their values without the offset are equal.

This means that grouping by a `TIMESTAMP_TZ` value will follow the same equality
rules above, and we make no guarantees about what the offset of the key for a
group will be - only guarantee is that the key's `UTC` timestamp is equal to all
values for that group. For example, consider the following table:

| A                          | B |
|----------------------------|---|
| 2023-01-01 00:00:00 +00:00 | 1 |
| 2023-01-01 01:00:00 +01:00 | 1 |
| 2023-01-01 00:00:00 +01:00 | 1 |
| 2023-01-01 01:00:00 +00:00 | 1 |
| 2023-01-02 00:00:00 +00:00 | 1 |
| 2023-01-02 01:00:00 +01:00 | 1 |

Where `A` is a `TIMESTAMP_TZ` and `B` is a `NUMBER`. Note that rows `0` and `1`
have equal values for `A`. Similarly rows `2` and `3` are equal in terms of `A`,
and same for rows `4` and `5`. Then, both of the following are valid results for
`SELECT A, sum(B) FROM table GROUP BY A`:

| A                          | B |
|----------------------------|---|
| 2023-01-01 00:00:00 +00:00 | 2 |
| 2023-01-01 00:00:00 +01:00 | 2 |
| 2023-01-02 00:00:00 +00:00 | 2 |

| A                          | B |
|----------------------------|---|
| 2023-01-01 01:00:00 +01:00 | 2 |
| 2023-01-01 00:00:00 +01:00 | 2 |
| 2023-01-02 01:00:00 +01:00 | 2 |

Note that these aren't the only two possibilities - for the query above there
are `8` possible results.

If you need to compare values by their local timestamp instead of their UTC
timestamp, consider casting to `timestampntz`. For the same input table above,
here's what the result of `SELECT A::timestampntz FROM table` would look like:

| A::timestampntz     |
|---------------------|
| 2023-01-01 00:00:00 |
| 2023-01-01 01:00:00 |
| 2023-01-01 00:00:00 |
| 2023-01-01 01:00:00 |
| 2023-01-02 00:00:00 |
| 2023-01-02 01:00:00 |


Note that this model of equality also holds during `JOIN`s:

Table 1:
| A                          | B |
|----------------------------|---|
| 2023-01-01 00:00:00 +00:00 | 1 |
| 2024-02-02 00:00:00 +00:00 | 2 |

Table 2:
| A                          |
|----------------------------|
| 2023-01-01 00:00:00 +01:00 |
| 2023-01-01 00:00:00 +02:00 |
| 2023-01-01 00:00:00 +03:00 |
| 2024-02-02 00:00:00 +01:00 |
| 2024-02-02 00:00:00 +02:00 |
| 2024-02-02 00:00:00 +03:00 |

The result of `SELECT TABLE1.A, TABLE2.A, B FROM TABLE1 JOIN TABLE2 ON TABLE1.A=TABLE2.A` would be:

| TABLE1.A                   | TABLE2.A                   | B |
|----------------------------|----------------------------|---|
| 2023-01-01 00:00:00 +00:00 | 2023-01-01 00:00:00 +01:00 | 1 |
| 2023-01-01 00:00:00 +00:00 | 2023-01-01 00:00:00 +02:00 | 1 |
| 2023-01-01 00:00:00 +00:00 | 2023-01-01 00:00:00 +03:00 | 1 |
| 2024-02-02 00:00:00 +00:00 | 2024-02-02 00:00:00 +01:00 | 2 |
| 2024-02-02 00:00:00 +00:00 | 2024-02-02 00:00:00 +02:00 | 2 |
| 2024-02-02 00:00:00 +00:00 | 2024-02-02 00:00:00 +03:00 | 2 |


Aside from comparison most other operations will treat `TIMESTAMP_TZ` as it's
local timestamp, for example `SELECT EXTRACT(HOUR from '2024-01-02 03:04:05 +06:07'::timestamptz)`
should return `3` (even though the `UTC` timestamp would have an hour of `21`).

#### TIMESTAMP\_TZ interaction with Snowflake

Note that reading `TIMESTAMP_TZ` values to or from Snowflake may change the
session parameter `TIMESTAMP_TZ_OUTPUT_FORMAT`. If your query relies on custom
values for `TIMESTAMP_TZ_OUTPUT_FORMAT` you may experience unexpected behavior.

#### TIMESTAMP\_TZ limitations

Currently only the following aggregation functions are supported with
`TIMESTAMP_TZ`. Future releases will expand this list.

+ min/max
+ first/last/any\_value
+ count
+ mode

Additionally, `TIMESTAMP_TZ` is *not* supported in semi-structured data (arrays,
 and objects).


## Supported Literals

BodoSQL supports the following literal types:

-   `#!sql array_literal`
-   `#!sql boolean_literal`
-   `#!sql datetime_literal`
-   `#!sql float_literal`
-   `#!sql integer_literal`
-   `#!sql interval_literal`
-   `#!sql object_literal`
-   `#!sql string_literal`
-   `#!sql binary_literal`

### Array Literal {#array_literal}

**Syntax**:

```sql
<[> [expr[, expr...]] <]>
```

where `<[>` and `<]>` indicate literal `[` and `]`s, and `expr` is any expression.

Array literals are lists of comma separated expressions wrapped in square brackets.

Note that BodoSQL currently only supports homogenous lists, and all `expr`s
must coerce to a single type.


### Boolean Literal {#boolean_literal}

**Syntax**:

```sql
TRUE | FALSE
```

Boolean literals are case-insensitive.

### Datetime Literal {#datetime_literal}

**Syntax**:

```sql
DATE 'yyyy-mm-dd' |
TIME 'HH:mm:ss' |
TIMESTAMP 'yyyy-mm-dd' |
TIMESTAMP 'yyyy-mm-dd HH:mm:ss'
```
### Float Literal {#float_literal}

**Syntax**:

```sql
[ + | - ] { digit [ ... ] . [ digit [ ... ] ] | . digit [ ... ] }
```

where digit is any numeral from 0 to 9

### Integer Literal {#integer_literal}

**Syntax**:

```sql
[ + | - ] digit [ ... ]
```

where digit is any numeral from 0 to 9

### Interval Literal {#interval_literal}

**Syntax**:

```sql
INTERVAL integer_literal interval_type
```

Where integer_literal is a valid integer literal and interval type is
one of:

```sql
DAY[S] | HOUR[S] | MINUTE[S] | SECOND[S]
```

In addition, we also have limited support for `#!sql YEAR[S]` and `#!sql MONTH[S]`.
These literals cannot be stored in columns and currently are only
supported for operations involving add and sub.


### Object Literal {#object_literal}

**Syntax**:

```sql
{['k1': `v1`[, 'k2': `v2`, ...]]}
```

Where each `ki` is a unique string literal, and each `vi` is an expression.
Obeys the same semantics as the function `#!sql OBJECT_CONSTRUCT` , so any pair
where the key or value is null is omitted, and for now BodoSQL only supports
when all values are the same type.


### String Literal {#string_literal}

**Syntax**:

```sql
'char [ ... ]'
```

Where char is a character literal in a Python string.


### Binary Literal {#binary_literal}

**Syntax**:

```sql
X'hex [ ... ]'
```

Where hex is a hexadecimal character between 0-F.
