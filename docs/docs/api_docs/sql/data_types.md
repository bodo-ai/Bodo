## Supported DataFrame Data Types

BodoSQL uses Pandas DataFrames to represent SQL tables in memory and
converts SQL types to corresponding Python types which are used by Bodo.
Below is a table mapping SQL types used in BodoSQL to their respective
Python types and Bodo data types.


<center>

| SQL Type(s)          | Equivalent Python Type | Bodo Data Type       |
|----------------------|------------------------|----------------------|
| `TINYINT`            | `np.int8`              | `bodo.int8`          |
| `SMALLINT`           | `np.int16`             | `bodo.int16`         |
| `INT`                | `np.int32`             | `bodo.int32`         |
| `BIGINT`             | `np.int64`             | `bodo.int64`         |
| `FLOAT`              | `np.float32`           | `bodo.float32`       |
| `DECIMAL`, `DOUBLE`  | `np.float64`           | `bodo.float64`       |
| `VARCHAR`, `CHAR`    | `str`                  | `bodo.string_type`   |
| `TIMESTAMP`, `DATE`  | `np.datetime64[ns]`    | `bodo.datetime64ns`  |
| `INTERVAL(day-time)` | `np.timedelta64[ns]`   | `bodo.timedelta64ns` |
| `BOOLEAN`            | `np.bool_`             | `bodo.bool_`         |

</center>

BodoSQL can also process DataFrames that contain Categorical or Date
columns. However, Bodo will convert these columns to one of the
supported types, which incurs a performance cost. We recommend
restricting your DataFrames to the directly supported types when
possible.

### Nullable and Unsigned Types

Although SQL does not explicitly support unsigned types, by default,
BodoSQL maintains the exact types of the existing DataFrames registered
in a [BodoSQLContext], including unsigned and non-nullable
type behavior. If an operation has the possibility of creating null
values or requires casting data, BodoSQL will convert the input of that
operation to a nullable, signed version of the type.

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

### Array Literal {#array_literal}

**Syntax**:

```sql
<[> [expr[, expr...]] <]>
```

where `<[>` and `<]>` indicate literal `[` and `]`s, and `expr` is any expression.

Array literals are lists of comma seperated expressions wrapped in square brackets.

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
