Reading and Writing Iceberg in Bodo {#iceberg_read_write}
=================


### SQL {#iceberg-sql}

BodoSQL can be used to read, create, or insert into an Iceberg table. Iceberg Tables are automatically detected by existing catalogs and are used during read:

- Snowflake Iceberg Tables are automatically detected when using the [`SnowflakeCatalog`][snowflake-catalog-api].
- Tables within the specified warehouse are automatically detected when using the [`RESTCatalog`][REST-catalog-api].
- Tables within the specified warehouse are automatically detected when using the [`GlueCatalog`][glue-catalog-api].
- Tables within the specified warehouse are automatically detected when using the [`S3TablesCatalog`][s3-tables-catalog-api].
- Hadoop Iceberg Catalogs and Tables are detected when using the [`FileSystemCatalog`][fs-catalog-api].
- Other Catalogs supported in the Python APIs can be accessed via the [`TablePath`][table-path-api] API using the same [connection string syntax][iceberg-conn-str].

To query an Iceberg table, use the standard `SELECT` syntax. To learn more about supported SELECT syntax, see the [SELECT][select] API reference.

```sql
SELECT ... FROM <... namespace_path ...>.<table_name> ...
```

#### Write Support

The `CREATE TABLE` syntax can be used to create Iceberg tables:

```sql
CREATE [OR REPLACE] [TRANSIENT | TEMPORARY] TABLE <...namespace_path...>.<table_name>
```

Inserting into existing Iceberg tables is supported via the `INSERT INTO` syntax:

```sql
INSERT INTO <...namespace_path...>.<table_name>
```

##### Snowflake Iceberg Write Support

To create Iceberg tables in Snowflake, a [Snowflake External Volume](https://docs.snowflake.com/en/user-guide/tables-iceberg-configure-external-volume) is required. The volume to use must be specified via the `exvol` argument to the `SnowflakeCatalog`:

```py
catalog = bodosql.SnowflakeCatalog(
    ...
    exvol='<... Snowflake Volume ...>'
)

bc = bodosql.BodoSQLContext(catalog=catalog)
```

!!! warning
    - Inserting into Snowflake Managed Iceberg Tables is not supported.
    - When the `exvol` parameter is specified, all tables constructed via `CREATE TABLE` will be Snowflake Iceberg tables.


### Python {#iceberg-python}


Bodo supports reading and writing to Iceberg tables from multiple catalogs and object stores (local, S3, and HDFS).

- Iceberg Reads are supported through the `pandas.read_sql_table` API.
- Iceberg Writes are supported through the `pandas.DataFrame.to_sql` API.


#### Connection String Syntax {#iceberg-conn-str}

To specify the Iceberg catalog in the Pandas APIs, the `conn` parameter must contain a connection string in one of the following formats.

Iceberg connection strings vary by catalog, but in general are of the form `iceberg<+conn>://<path><?params>` where
- `<conn>://<path>` is the location of the catalog or Iceberg warehouse
- `params` is a list of properties to pass to the catalog. Each parameter must be of the form `<key>=<value>` and separated with `&`, similar to HTTP URLs.

The following parameters are supported:
- `type`: Type of catalog. The supported values are listed below. When the connection string is ambiguous, this parameter is used to determine the type of catalog implementation.
- `warehouse`: Location of the warehouse. Required when creating a new table using a Glue or Hive catalog.

The following catalogs are supported:

- Hadoop Catalog on Local Filesystem:
    - Used when `type=hadoop` is specified **or** when `<conn>` is `file` or empty
    - `<path>` is the absolute path to the warehouse (directory containing the database schema)
    - Parameter `warehouse` will be ignored if specified
    - E.g. `iceberg://<ABSOLUTE PATH TO ICEBERG WAREHOUSE>` or `iceberg+file://<ABSOLUTE PATH TO ICEBERG WAREHOUSE>`

- Hadoop Catalog on S3
    - Used when `type=hadoop-s3` is specified **or** when `<conn>` is `s3`.
    - `<conn>://<path>` is the S3 path to the warehouse (directory or bucket containing the database schema).
    - Parameter `warehouse` will be ignored if specified.
    - E.g. `iceberg+s3://<S3 PATH TO ICEBERG WAREHOUSE>`

- AWS Glue Catalog
    - Connection string must be of the form `iceberg+glue?<params>`.
    - Parameters `glue.id` (account id) and `glue.region` can be used to specify different catalogs.
    - Parameter `type` will be ignored if specified.
    - Parameter `warehouse` is required to create a table.
    - E.g. `iceberg+glue` or `iceberg+glue?warehouse=s3://<ICEBERG-BUCKET>`

- S3 Tables Catalog
    - Connection string must be of the form `iceberg+arn:aws:s3tables:<region>:<account_number>:bucket/<bucket>`
    - `params` is unused
    - E.g. `iceberg+arn:aws:s3tables:<region>:<account_number>:bucket/<bucket>`

- Hive / Thrift Catalog
    - Used when `type=hive` is specified **or** when `<conn>` is `thrift`.
    - `<conn>://<path>` is the URL to the Thrift catalog, i.e. `thrift://localhost:9083`.
    - Parameter `warehouse` is required to create the table.
    - E.g. `iceberg+thrift://<THRIFT URL>`

- REST Catalog
    - Connection string must be of the form `iceberg+http(s)://<catalog_host>?<params>`.
    - Parameter `type` will be ignored if specified.
    - Parameter `warehouse` is required.
    - Parameter `token` or `credential` is required for authentication and should be retrieved from the REST catalog provider.
    - E.g.`iceberg+http(s)://<rest-uri>?warehouse=<warehouse>&token=<token>`

- S3 Tables
    - Connection string must be of the form `iceberg+arn:aws:s3tables:<region>:<account_number>:bucket/<bucket>`
    - `params` is unused
    - E.g. `iceberg+arn:aws:s3tables:us-west-2:123456789012:bucket/mybucket`

#### Pandas APIs {#iceberg-pandas}

Example code for reading:

```py
@bodo.jit
def example_read_iceberg() -> pd.DataFrame:
    return pd.read_sql_table(
        table_name="<... Name of the Iceberg Table ...>",
        con="<... Connection String. See previous section ...>",
        schema="<... Namespace Path to Iceberg Table ...>",
        _snapshot_id="<... Optional (int): Snapshot ID to read the table from ...>",
        _snapshot_timestamp_ms="<... Optional (int): Timestamp to read the table from ...>"

    )
```

!!! note
    - The `schema` argument is required for reading Iceberg tables.

    - The Iceberg table to read should be located at `<warehouse-location>/<schema>/<table_name>`,
      where `schema` and `table_name` are the arguments to `pd.read_sql_table`, and `warehouse-location`
      is inferred from the connection string based on the description provided above.

An example for writing to Iceberg via `pandas.DataFrame.to_sql`:

```py
@bodo.jit(distributed=["df"])
def write_iceberg_table(df: pandas.DataFrame):
    df.to_sql(
        name="<... Name of the Iceberg Table ...>",
        con="<... Connection String. See previous section ...>",
        schema="<... Namespace Path to Iceberg Table ..>",
        if_exists="replace"
    )
```

!!! note
    - `schema` argument is required for writing Iceberg tables.
    - Writing a Pandas Dataframe index to an Iceberg table is not supported. If `index` and `index_label`
      are provided, they will be ignored.
    - `chunksize`, `dtype` and `method` arguments are not supported and will be ignored if provided.

#### Table Partitioning and Sorting {#iceberg-partitioning-sorting}
Bodo supports reading and writing Iceberg tables with partitioning and sorting. Bodo doesn't support creating a new table with partitioning or sorting yet. We recommend using PyIceberg to create a new empty table with partitioning and sorting specified and using Bodo for read and write of actual data. PyIceberg is a dependency of Bodo and is automatically installed when you install Bodo.

Example:
```py
import pandas as pd
from pyiceberg import schema, types
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.transforms import BucketTransform, DayTransform

import bodo
from bodo.io.iceberg.catalog import conn_str_to_catalog

# Establish a connection to the Iceberg catalog
catalog = conn_str_to_catalog("iceberg+file://iceberg_db")

# Define the schema for the Iceberg table
schema = schema.Schema(
        types.NestedField(0, "id", types.LongType()),
        types.NestedField(1, "timestamp", types.TimestamptzType()),
        types.NestedField(2, "data", types.StringType()),  #
)


# Create the Iceberg table if it doesn't already exist
if not catalog.table_exists("default.test_table"):
    # Specify partitioning: bucket by ID and partition by day for timestamp
    partition_spec = PartitionSpec(
                PartitionField(
                        transform=BucketTransform(3),  # Bucketing with 3 buckets
                        source_id=0,
                        field_id=1000,
                        name="bucket",
                    ),
                PartitionField(
                        transform=DayTransform(),  # Partitioning by day
                        source_id=1,
                        field_id=1001,
                        name="day",
                    ),
            )
        table = catalog.create_table(
                "default.test_table",
                schema=schema,
                partition_spec=partition_spec,
            )

@bodo.jit()
def append(df):
    # Append a DataFrame to the Iceberg table
    df.to_sql("test_table", schema="default", con="iceberg+file://iceberg_db", if_exists="append")

# Example data to append: two rows with IDs, timestamps, and string data
append(pd.DataFrame({
        "id": [1, 2],
        "timestamp": [pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(3, "days")],
        "data": ["a",]
}))
```
