Supported Iceberg Data Types {#iceberg_dtypes}
=================

Bodo supports most data types defined in the Apache Iceberg specification. This following table shows how Iceberg data types are represented in Python and SQL.

| Iceberg Data Type | Equivalent Python / Pandas Array Type | Equivalent SQL Column Type |
|-------------------|---------------------------------------|----------------------------|
| boolean           | `bool[pyarrow]`                       | BOOL                       |
| int               | `int32[pyarrow]`                      | INT                        |
| long              | `int64[pyarrow]`                      | BIGINT                     |
| float             | `float32[pyarrow]`                    | FLOAT                      |
| double            | `float64[pyarrow]`                    | DOUBLE                     |
| decimal(P, S)     | `decimal128(P, S)[pyarrow]`           | DECIMAL(P, S)              |
| date              | `date32[pyarrow]`                     | DATE                       |
| time              | `time32[pyarrow]`                     | TIME                       |
| timestamp         | `timestamp[us][pyarrow]`              | TIMESTAMP                  |
| timestamptz       | `timestamp[us, tz=UTC][pyarrow]`      | TIMESTAMPTZ                |
| string            | `large_string[pyarrow]`               | STRING                     |
| binary            | `binary[pyarrow]`                     | BINARY                     |
| struct<...>       | `struct<...>[pyarrow]`                | STRUCT                     |
| list<E>           | `large_list<E>[pyarrow]`              | LIST                       |
| map<K, V>         | `map<K, V>[pyarrow]`                  | MAP                        |

Bodo does not support these Iceberg data types yet:

- fixed
- uuid
