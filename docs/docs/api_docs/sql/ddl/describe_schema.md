# DESCRIBE SCHEMA

Describes the schema. For example, lists the tables and views in the schema.

For Iceberg catalogs, lists the tables and views of the namespace.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/desc-schema) for more details.

## Syntax

```sql
DESC[RIBE] SCHEMA <name>
```

## Output

The output provides object properties and metadata in the following columns:

| Column | DESCRIPTION |
|:-------------|:---------------|
| CREATED_ON | The timestamp at which the object was created, **as a string type**. |
| NAME | The name of the object. |
| KIND | The kind of the object. |

For Iceberg tables, the `CREATED_ON` field will be `None`.

## Examples

```sql
DESCRIBE SCHEMA sample_schema;

+-------------------------------+----------------+-------------------+
| CREATED_ON                    | NAME           | KIND              |
|-------------------------------+----------------+-------------------|
| 2022-06-23 01:00:00.000 -0700 | SAMPLE_TABLE_1 | TABLE             |
| 2022-06-23 02:00:00.000 -0700 | SAMPLE_VIEW_1  | VIEW              |
+-------------------------------+----------------+-------------------+
```
