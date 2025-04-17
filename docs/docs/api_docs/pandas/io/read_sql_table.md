# pd.read_sql_table

`pandas.read_sql_table`

-   This API only supports reading Iceberg tables at the moment.
-   See the [Iceberg Section][iceberg_read_write] for example usage and more system specific instructions.
-   Argument `table_name` is supported and must be the name of an Iceberg Table.
-   Argument `con` is supported but only as a string form in a URL format.
    SQLalchemy `connectable` is not supported.
    It should be the absolute path to a Iceberg warehouse.
    If using a Hadoop-based directory catalog, it should start with the URL scheme `iceberg://`.
    If using a Thrift Hive catalog, it should start with the URL scheme `iceberg+thrift://`
-   Argument `schema` is supported and currently required for Iceberg tables. It must be the name
    of the database schema. For Iceberg Tables, this is the directory name
    in the warehouse (specified by `con`) where your table exists.
-   Arguments `index_col`, `coerce_float`, `parse_dates`, `columns` and `chunksize` are
    not supported.
-   Arguments `_snapshot_id` and `_snapshot_timestamp_ms` are only available for Iceberg tables. These Arguments
    are experimental and may change without warning. These arguments may be used to read a table from
    a specific snapshot or point in time, which is known as "time travel" in Iceberg.

