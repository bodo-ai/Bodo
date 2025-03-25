# Database Catalogs

Database Catalogs are configuration objects that grant BodoSQL access to load tables from a database.
For example, when a user wants to load data from Snowflake, a user will create a `SnowflakeCatalog` to grant
BodoSQL access to their Snowflake account and load the tables of interest.

A database catalog can be registered during the construction of the `BodoSQLContext` by passing it in as a parameter, or can be manually set using the
`BodoSQLContext.add_or_replace_catalog` API. Currently, a `BodoSQLContext` can support at most one database catalog.

When using a catalog in a `BodoSQLContext` we strongly recommend creating the `BodoSQLContext` once in regular Python and then
passing the `BodoSQLContext` as an argument to JIT functions. There is no benefit to creating the
`BodoSQLContext` in JIT and this could increase compilation time.

```py
catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account_name,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # database name
)
bc = bodosql.BodoSQLContext({"LOCAL_TABLE1": df1}, catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name, local_id FROM TPCH_SF1.REGION, local_table1 WHERE R_REGIONKEY = local_table1.region_key ORDER BY r_name")

run_query(bc)
```

Database catalogs can be used alongside local, in-memory `DataFrame` or `TablePath` tables. If a table is
specified without a schema then BodoSQL resolves the table in the following order:

1. Default Catalog Schema
2. Local (in-memory) DataFrames / TablePath names

An error is raised if the table cannot be resolved after searching through both of these data sources.

This ordering indicates that in the event of a name conflict between a table in the database catalog and a local table, the table in the database catalog is used.

If a user wants to use the local table instead, the user can explicitly specify the table with the local schema `__BODOLOCAL__`.

For example:

```SQL
SELECT A from __BODOLOCAL__.table1
```

Currently, BodoSQL supports catalogs for Snowflake and a user's FileSystem. Support for other data storage systems will be added in future releases.

## SnowflakeCatalog {#snowflake-catalog-api}

With a Snowflake Catalog, users only have to specify their Snowflake connection once.
They can then access any tables of interest in their Snowflake account.
Currently, a Snowflake Catalog requires a default `DATABASE` (e.g., `USE DATABASE`), as shown below.

```py

catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account_name,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # default database name
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

run_query(bc)
```

BodoSQL does not currently support Snowflake syntax for specifying defaults
and session parameters (e.g. `USING SCHEMA <NAME>`). Instead users can pass
any session parameters through the optional `connection_params` argument, which
accepts a `Dict[str, str]` for each session parameter. For example, users can provide
a default schema to simplify the previous example.

```py

catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # database name
    connection_params={"schema": "TPCH_SF1"}
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name FROM REGION ORDER BY r_name")

run_query(bc)
```

Internally, Bodo uses the following connections to Snowflake:

1. A JDBC connection to lazily fetch metadata.
2. The Snowflake-Python-Connector's distributed fetch API to load batches of arrow data.

### API Reference

- `bodosql.SnowflakeCatalog(username: str, password: str, account: str, warehouse: str, database: str, connection_params: Optional[Dict[str, str]] = None, iceberg_volume: Optional[str] = None)`
<br><br>

    Constructor for `SnowflakeCatalog`. This allows users to execute queries on tables stored in Snowflake when the `SnowflakeCatalog` object is registered with a `BodoSQLContext`.

    ***Arguments***

    - `username`: Snowflake account username.

    - `password`: Snowflake account password.

    - `account`: Snowflake account name.

    - `warehouse`: Snowflake warehouse to use when loading data.

    - `database`: Name of Snowflake database to load data from. The Snowflake
        Catalog is currently restricted to using a single Snowflake `database`.

    - `connection_params`: A dictionary of Snowflake session parameters.

    - `iceberg_volume`: The name of a storage volume to use for writing Iceberg tables. When provided any tables created by BodoSQL will be written as
       an Iceberg table.


#### Supported Query Types

The `SnowflakeCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql INSERT INTO`
  * `#!sql DELETE`
  * `#!sql CREATE TABLE AS`

## FileSystemCatalog {#fs-catalog-api}

The `FileSystemCatalog` allows users to read and write tables using their local file system or S3 storage
without needing access to a proper database. To use this catalog, you will have to select a root directory.
The catalog will treat each subdirectory as a schema that you can also specify. We recommend always
using at least one schema to avoid any potential issues with table resolution. For example, the following code shows
how a user could read a table called `MY_TABLE` that is located at `s3://my_bucket/MY_SCHEMA/MY_TABLE`.

```py
catalog = bodosql.FileSystemCatalog(
    "s3://my_bucket", # root directory
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT * FROM MY_SCHEMA.MY_TABLE")

run_query(bc)
```

When working with tables in the `FileSystemCatalog`, BodoSQL uses the full name of any
directory or file as the object's name and is case sensitive. When constructing a query
you must follow the BodoSQL rules for [identifier case sensitivity][identifier-case-sensitivity].

To simplify your queries you can also provide a default schema resolution path to the `FileSystemCatalog` constructor.
For example, this code provides a default schema of `MY_SCHEMA.other_schema` for loading `OTHER_TABLE` from
`s3://my_bucket/MY_SCHEMA/other_schema/OTHER_TABLE`.

```py
catalog = bodosql.FileSystemCatalog(
    "s3://my_bucket",
    default_schema="MY_SCHEMA.\"other_schema\""
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT * FROM OTHER_TABLE")

run_query(bc)
```

### API Reference

- `bodosql.FileSystemCatalog(root: str, default_write_format: str = "iceberg", default_schema: str = ".")`
<br><br>

    Constructor for `FileSystemCatalog`. This allows users to try a file system as a database for querying
    or writing tables with a `BodoSQLContext`.

    ***Arguments***

    - `root`: Filesystem path that provides the root directory for the database. This can either be a local file system path or an S3 path.

    - `default_write_format`: The default format to use when writing tables using `#!sql create table as`. This can be either `iceberg` or `parquet`.

    - `default_schema`: The default schema to use when resolving tables. This should be a `.` separated string that represents the path to the default schema.
       Each value separated by a `.` should be treated as its own SQL identifier. If no default schema is provided the root directory is used.

#### Supported Query Types

The `FileSystemCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql CREATE TABLE AS`


#### Supported Table Types

The `FileSystemCatalog` currently only supports reading Iceberg tables. It can write tables as either Iceberg or Parquet,
depending on the `default_write_format` parameter. When writing tables, any specified schema must already exist as directories
in the file system. Future releases will provide additional table support.

#### S3 Support

The `FileSystemCatalog` supports reading and writing tables from S3. When using S3, the `root` parameter should be an s3 uri.
To access S3 BodoSQL uses the following environment variables to connect to S3:

  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_REGION`

If you encounter any issues connecting to s3 or accessing a table, please ensure that these environment variables are set.
For more information please refer to the [AWS documentation.](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html)

## RESTCatalog {#REST-catalog-api}

The `RESTCatalog` allows users to read and write tables to and from REST Iceberg catalogs.
To use this catalog, you will need to provide the uri of the rest catalog and a token or credential.

```py
catalog = bodosql.RESTTablesCatalog(
    warehouse="warehouse_name",
    rest_uri="http://rest_uri",
    token="token",
)
bc = bodosql.BodoSQLContext(catalog=catalog)
df = bc.sql("SELECT * FROM MY_SCHEMA.MY_TABLE")
```

When constructing a query you must follow the BodoSQL rules for [identifier case sensitivity][identifier-case-sensitivity].
### API Reference

``` python
bodosql.RESTCatalog(
    warehouse: str,
    rest_uri: str,
    token: str | None = None,
    credential: str | None = None,
    scope: str | None = None,
    default_schema: str | None = None)
```
    ***Arguments***

    - `warehouse`: Name of the REST Iceberg catalog warehuose to connect to
    - rest_uri: URI of the REST Iceberg catalog to connect to
    - token: Token to use for authentication, if credential is not provided
    - credential: Credential to use for authentication, if token is not provided
    - scope: Scope to authenticate with, not always required
    - default_schema: Default schema to use when resolving tables. If not provided, the root schema is used.

#### Supported Query Types

The `RESTCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql CREATE TABLE AS`


## GlueCatalog {#glue-catalog-api}

The `GlueCatalog` allows users to read and write tables using AWS Glue.
To use this catalog, you will have to select a warehouse that is the Glue S3 bucket.
The bucket must exist. The `s3://` prefix is optional. I.e., both `s3://bucket_name` and `bucket_name` are valid.

```py
catalog = bodosql.GlueCatalog(
    warehouse="warehouse_name",
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT * FROM MY_SCHEMA.MY_TABLE")

run_query(bc)
```

When constructing a query you must follow the BodoSQL rules for [identifier case sensitivity][identifier-case-sensitivity].

### Authentication / Authorization

Before creating a catalog for the Glue S3 bucket, you must first ensure that the cloud config role for the workspace has access to the Glue S3 bucket with the following permissions:
```
{
    "Sid": "BodoPlatformCatalog",
    "Effect": "Allow",
    "Action": [
        "s3:ListBucket"
    ],
    "Resource": "arn:aws:s3:::<backet>"
}
```

The following are the steps to find the cloud config role:
  * find the `Cloud Config uuid` in the workpace details.
  * find the `Cloud Config` that has the same uuid in the `Cloud Configurations` page.
  * find the `Role ARN` in the details of the `Cloud Config`. It may look like `BodoPlatformUser-XXXXXXXX`
  if it was created using `Cloud Formation` or `Access Key`. 

The Bodo clusters that run the queries on the Glue Catalog need to be created with an instance role. This instance role needs to have read/write access to
the Glue S3 bucket and also appropreate Glue permissions. The exact set of Glue permissions depend on the queries on the Glue database. The following is 
an example that could work for most of the use cases.

```
{
    "Sid": "BodoPlatformAccessGlueCatalog",
    "Effect": "Allow",
    "Action": [
        "glue:CreateDatabase",
        "glue:CreateSchema",
        "glue:CreateTable",
        "glue:DeleteDatabase",
        "glue:DeleteSchema",
        "glue:DeleteTable",
        "glue:GetDatabase",
        "glue:GetDatabases",
        "glue:GetSchema",
        "glue:GetSchemaByDefinition",
        "glue:GetSchemaVersion",
        "glue:GetSchemaVersionsDiff",
        "glue:GetTable",
        "glue:GetTables",
        "glue:GetTableVersion",
        "glue:GetTableVersions",
        "glue:ListSchemas",
        "glue:ListSchemaVersions",
        "glue:PutSchemaVersionMetadata",
        "glue:QuerySchemaVersionMetadata",
        "glue:RegisterSchemaVersion",
        "glue:RemoveSchemaVersionMetadata",
        "glue:UpdateDatabase",
        "glue:UpdateSchema",
        "glue:UpdateTable"
    ],
    "Resource": [
      "arn:aws:glue:*:*:catalog",
      "arn:aws:glue:*:*:database/*<bodo-db>*",
      "arn:aws:glue:*:*:table/*<bodo-db>*/<table>",
      "arn:aws:glue:*:*:connection/*<bodo-connection>*",
      "arn:aws:glue:*:*:session/*<bodo-session>*"
    ]
}, {
    "Sid": "BodoPlatformAccessGlueBucket",
    "Effect": "Allow",
    "Action": [
        "s3:*",
    ],
   "Resource": "arn:aws:s3:::<backet>"
}
```

For workspaces that have PrivateLink enabled, or have no internet access, users need to create AWS Glue endpoint in the VPC to allow access to Glue.

### API Reference

- `bodosql.GlueCatalog(warehouse: str)`
<br><br>

    Constructor for `GlueCatalog`. This allows users to use an AWS Glue Iceberg Warehouse as a database for querying
    or writing tables with a `BodoSQLContext`.

    ***Arguments***

    - `warehouse`: Name of the Glue S3 bucket, with or without the `s3://` prefix. I.e., both `s3://bucket_name` and `bucket_name` are valid.

#### Supported Query Types

The `GlueCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql CREATE TABLE AS`


## S3TablesCatalog {#s3-tables-catalog-api}

The `S3TablesCatalog` allows users to read and write tables to and from S3 Tables.
To use this catalog, you will need to provide the S3 table bucket arn.
The bucket must exist.

```py
catalog = bodosql.S3TablesCatalog(
    warehouse="warehouse_name",
)
bc = bodosql.BodoSQLContext(catalog=catalog)
df = bc.sql("SELECT * FROM MY_SCHEMA.MY_TABLE")
```

When constructing a query you must follow the BodoSQL rules for [identifier case sensitivity][identifier-case-sensitivity].

### Authentication / Authorization

Refer to [AWS' documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-tables-security-overview.html)
### API Reference

- `bodosql.S3TablesCatalog(warehouse: str)`
<br><br>

    Constructor for `S3TablesCatalog`. This allows users to use an AWS S3 Tables Iceberg Warehouse as a database for querying
    or writing tables with a `BodoSQLContext`.

    ***Arguments***

    - `warehouse`: Arn of the S3 table bucket

#### Supported Query Types

The `S3TablesCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql CREATE TABLE AS`

