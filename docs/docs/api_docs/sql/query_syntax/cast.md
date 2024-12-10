# CAST



THE `#!sql CAST` operator converts an input from one type to another. In
many cases casts are created implicitly, but this operator can be
used to force a type conversion.

The following casts are currently supported. Please refer to
`supported_dataframe_data_types` for
the Python types for each type keyword:

| From                             | To                              | Notes                                                                         |
|----------------------------------|---------------------------------|-------------------------------------------------------------------------------|
|`VARCHAR`                         |`VARCHAR`                        |                                                                               |
|`VARCHAR`                         |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`VARCHAR`                         |`FLOAT/DOUBLE`                   |                                                                               |
|`VARCHAR`                         |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future.                        |
|`VARCHAR`                         |`TIMESTAMP`                      |                                                                               |
|`VARCHAR`                         |`DATE`                           | Truncates to date but is still Timestamp type. This may change in the future. |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`VARCHAR`                        |                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`FLOAT/DOUBLE`                   |                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future.                        |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`TIMESTAMP`                      |                                                                               |
|`FLOAT/DOUBLE`                    |`VARCHAR`                        |                                                                               |
|`FLOAT/DOUBLE`                    |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`FLOAT/DOUBLE`                    |`FLOAT/DOUBLE`                   |                                                                               |
|`FLOAT/DOUBLE`                    |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future                         |
|`TIMESTAMP`                       |`VARCHAR`                        |                                                                               |
|`TIMESTAMP`                       |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`TIMESTAMP`                       |`TIMESTAMP`                      |                                                                               |
|`TIMESTAMP`                       |`DATE`                           | Truncates to date but is still Timestamp type. This may change in the future. |

!!! note
    `#!sql CAST` correctness can often not be determined at compile time.
    Users are responsible for ensuring that conversion is possible
    (e.g. `#!sql CAST(str_col as INTEGER)`).

