BodoSQL {#bodosql}
========

BodoSQL provides high performance and scalable SQL query execution
using Bodo's HPC capabilities and optimizations. It also provides
native Python/SQL integration as well as SQL to Pandas conversion for
the first time.



## Aliasing

In all but the most trivial cases, BodoSQL generates internal names to
avoid conflicts in the intermediate dataframes. By default, BodoSQL
does not rename the columns for the final output of a query using a
consistent approach. For example the query:

```sql
bc.sql("SELECT SUM(A) FROM table1 WHERE B > 4")
```
Results in an output column named `$EXPR0`. To reliably reference this
column later in your code, we highly recommend using aliases for all
columns that are the final outputs of a query, such as:

```py
bc.sql("SELECT SUM(A) as sum_col FROM table1 WHERE B > 4")
```

!!! note
     BodoSQL supports using aliases generated in `#!sql SELECT` inside
    `#!sql GROUP BY` and `#!sql HAVING` in the same query, but you cannot do so with
    `#!sql WHERE`

## User Defined Functions (UDFs) and User Defined Table Functions (UDTFs)

BodoSQL supports using Snowflake UDFs and UDTFs in queries and views. To make
UDFs and UDTFs available in BodoSQL, you must first register and define them inside
your Snowflake account using the appropriate `#!sql create function` command. Once
the function is created, so long as your user can access the
[function's metadata](https://docs.snowflake.com/en/sql-reference/info-schema/functions),
BodoSQL can process queries that use the function.

### Usage

A UDF is used like any other SQL function, except that there are two possible
calling conventions.

`#!sql MY_UDF(arg1, arg2, ..., argN)`

`#!sql MY_UDF(name1=>arg1, name2=>arg2, ..., nameN=>argN)`

When calling a function you must either pass all arguments positionally or
by name (you cannot mix these). If you pass the arguments by name, then you
can pass them in any order. For example, the following calls are are equivalent.

```sql
select my_udf(name1=>1, name2=>2) as A, my_udf(name2=>2, name1=>1) as B
```

When calling a UDTF you must wrap the function in a `#!sql TABLE()` call and then
you may use the function anywhere a table can be used. For example:

```sql
select * from table(my_udtf(1))
```

To reference columns from another table in the UDTF, you can use a comma join, optionally
alongside the `#!sql lateral` keyword. For example:

```sql
select * from my_table, table(my_udtf(N=>A))
```

or

```sql
select * from my_table, LATERAL(table(my_udtf(N=>A)))
```

### Calling Convention Best Practices

When calling either a UDF or a UDTF, we strongly recommend always using the named calling
convention. This is because UDFs support overloaded definitions and using distinct names
is the safest way to ensure you are calling the correct function. For more information see
this section of the [Snowflake Documentation](https://docs.snowflake.com/en/developer-guide/udf-stored-procedure-naming-conventions#overloading-procedures-and-functions).
Even if you are not currently using an overloaded function, we encourage this practice in case
the function is overloaded in the future.

### Requirements

BodoSQL must be able to execute the UDF directly from its definition. To do this,
BodoSQL needs to be able to both obtain the definition and execute it,
producing the following requirements:

- The function must be written in SQL.
- All elements of the function body must be supported within BodoSQL.
- The user executing Bodo must have access to any tables or views referenced
  within the function body.
- The function must not be defined using the secure keyword.
- The function must not be defined using the external keyword.

In addition, there are a couple other limitations to be aware of due to gaps in
the available metadata:

- At this time, we cannot support default values because the default is not stored in
  the metadata. These functions can still be executed by providing the default values.
- Some special characters in argument names, especially commas or spaces, may not compile
  because they are not properly escaped within the Snowflake metadata.


### Performance

BodoSQL supports UDFs and UDTFs by inlining the function body directly into the
body of the query. This means that users of these functions should achieve the same
performance as if they had written the function body directly into the query.

For complex UDFs or UDTFs, naively executing the function body may require producing a correlated
subquery, an operation in which a query must be executed once per row in another table.
This can cause a significant performance hit, so BodoSQL undergoes a process called
decorrelation to rewrite the query in terms of much more efficient joins. If BodoSQL
is not able to rewrite a query, then it will raise an error indicating a correlation
could not be fully removed.


### Overloaded Definition Priority

As mentioned above, Snowflake UDFs support overloaded definitions. This means that you can define
the same function name multiple times with different argument signatures,
and a function will be selected by determining the "best match", possibly through implicit casting.

BodoSQL supports this functionality, but if there is no exact match, then BodoSQL **cannot guarantee**
equivalent Snowflake behavior. Snowflake states which
[implicit casts are legal](https://docs.snowflake.com/en/sql-reference/data-type-conversion#data-types-that-can-be-cast),
but it provides no promises as to which function will be selected in the case of multiple
possible matches requiring implicit casts.

When BodoSQL encounters a UDF call, without an exact match, we look at the implicit cast priority of each
possible UDF defintions as shown in the table below.

<center>

| Source Type | Target Option 1 | Target Option 2 | Target Option 3 | Target Option 4 | Target Option 5 | Target Option 6 | Target Option 7 | Target Option 8 | Target Option 9 | Target Option 10 |
|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|
| ARRAY | VARIANT | | | | | | | | | |
| BOOLEAN | VARCHAR | VARIANT | | | | | | | | |
| DATE | TIMESTAMP_LTZ | TIMESTAMP_NTZ | VARCHAR | VARIANT | | | | | | |
| DOUBLE | BOOLEAN | VARIANT | VARCHAR | NUMBER | | | | | | |
| NUMBER | DOUBLE | BOOLEAN | VARIANT | VARCHAR | | | | | | |
| OBJECT | VARIANT | | | | | | | | | |
| TIME | VARCHAR | | | | | | | | | |
| TIMESTAMP_NTZ | TIMESTAMP_LTZ | VARCHAR | DATE | TIME | VARIANT | | | | | |
| TIMESTAMP_LTZ | TIMESTAMP_NTZ | VARCHAR | DATE | TIME | VARIANT | | | | | |
| VARCHAR | BOOLEAN | DATE | DOUBLE | TIMESTAMP_LTZ | TIMESTAMP_NTZ | NUMBER | TIME | VARIANT | | |
| VARIANT | ARRAY | BOOLEAN | OBJECT | VARCHAR | DATE | TIME | TIMESTAMP_LTZ | TIMESTAMP_NTZ | DOUBLE | NUMBER |

</center>

Here, the lower the option number, the higher the priority, with exact matches having priority 0 and being omitted.
If there is no function with an exact match then we compute the closest signature by computing the "priority" of the
required cast for each argument based on the above table and selecting the implementation with the smallest sum of distances.
If we encounter a tie then we select the earliest defined function based on the metadata. While this may not match Snowflake
in all situations, we have found that in common cases (e.g., differing by a single argument), this gives us behavior consistent with Snowflake.

However, as we add further type support or expand our UDF infrastructure, this matching system is subject to change. As a result,
we strongly recommend using a unique name for each argument and only using the named calling convention to avoid any potential issues.

## BodoSQL Caching & Parameterized Queries {#bodosql_named_params}

BodoSQL can reuse Bodo caching to avoid recompilation when used inside a
JIT function. BodoSQL caching works the same as Bodo, so for example:

```py
@bodo.jit(cache=True)
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"TABLE1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B > 4")
    print(df2.A.sum())
```

This will avoid recompilation so long as the DataFrame scheme stored in
`filename` has the same schema and the code does not change.

To enable caching for queries with scalar parameters that you may want
to adjust between runs, we introduce a feature called parameterized
queries. In a parameterized query, the SQL query replaces a
constant/scalar value with a variable, which we call a named parameter.
In addition, the query is passed a dictionary of parameters which maps
each name to a corresponding Python variable.

For example, if in the above SQL query we wanted to replace 4 with other
integers, we could rewrite our query as:

```py
bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
```

Now anywhere that `@var` is used, the value of python_var at runtime
will be used instead. This can be used in caching, because python_var
can be provided as an argument to the JIT function itself, thus enabling
changing the filter without recompiling. The full example looks like
this:

```py
@bodo.jit(cache=True)
def f(filename, python_var):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"TABLE1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
    print(df2.A.sum())
```

Named parameters cannot be used in places that require a constant value
to generate the correct implementation (e.g. TimeUnit in EXTRACT).

!!! note
    Named parameters are case sensitive, so `@var` and `@VAR` are
    different identifiers.

## IO Handling

BodoSQL is great for compute based SQL queries, but you cannot yet access external storage directly from SQL. Instead, you can load and store data using Bodo and various Python APIs. Here we explain a couple common methods for loading data.

### Pandas IO in JIT function with SQL Query

The most common way to load data is to first use Pandas APIs to load a DataFrame inside a JIT function and then to use that DataFrame inside a BodoSQLContext.

```py
def f(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    bc = bodosql.BodoSQLContext(
        {
            "T1": df1,
            "T2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```


### Pandas IO in a JIT Function Separate from Query

The previous approach works well for most individual queries. However, when running several queries on the same dataset, it should ideally be loaded once for all queries. To do this, you can structure your JIT code to contain a single load function at the beginning. For example:

```py

@bodo.jit
def load_data(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    return df1, df2

def q1(df1, df2):
    bc = bodosql.BodoSQLContext(
        {
            "T1": df1,
            "T2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")

...

@bodo.jit
def run_queries(f1, f2):
    df1, df2 = load_data(f1, f2)
    print(q1(df1, df2))
    print(q2(df2))
    print(q3(df1))
    ...

run_queries(f1, f2)
```

This approach prevents certain optimizations, such as filter pushdown. However, the assumption here is that you will use the entire DataFrame across the various benchmarks, so no optimization is useful by itself. In addition, any optimizations that can apply to all queries can be done explicitly inside `load_data`. For example, if all queries are operate on a single day's data with `df1`, you can write that filter in `load_data` to limit IO and filter pushdown will be performed.

```py

@bodo.jit
def load_data(f1, f2, target_date):
    df1 = pd.read_parquet(f1)
    # Applying this filter limits how much data is loaded.
    df1 = df1[df1.date_val == target_date]
    df2 = pd.read_parquet(f2)
    return df1, df2

@bodo.jit
def run_queries(f1, f2, target_date):
    df1, df2 = load_data(f1, f2, target_date)
    ...

run_queries(f1, f2, target_date)
```

## BodoSQLContext API

The `BodoSQLContext` API is the primary interface for executing SQL queries. It performs two roles:

  1. Registering data and connection information to load tables of interest.
  2. Forwarding SQL queries to the BodoSQL engine for compilation and execution. This is done via the
     `bc.sql(query)` method, where `bc` is a `BodoSQLContext` object.



A `BodoSQLContext` can be defined in regular Python and passed as an argument to JIT functions or can be
defined directly inside JIT functions. We recommend defining and modifying a `BodoSQLContext` in regular
Python whenever possible.

For example:

```py
bc = bodosql.BodoSQLContext(
    {
        "T1": bodosql.TablePath("my_file_path.pq", "parquet"),
    },
    catalog=bodosql.SnowflakeCatalog(
        username,
        password,
        account_name,
        warehouse_name,
        database name,
    )
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, catalogSchema.t2 where t1.C > 5 and t1.D = catalogSchema.t2.D")
```

### API Reference

- `bodosql.BodoSQLContext(tables: Optional[Dict[str, Union[pandas.DataFrame|TablePath]]] = None, catalog: Optional[DatabaseCatalog] = None)`
<br><br>

    Defines a `BodoSQLContext` with the given local tables and catalog.

    ***Arguments***

    - `tables`: A dictionary that maps a name used in a SQL query to a `DataFrame` or `TablePath` object.

    - `catalog`: A `DatabaseCatalog` used to load tables from a remote database (e.g. Snowflake).


- `bodosql.BodoSQLContext.sql(self, query: str, params_dict: Optional[Dict[str, Any] = None)`
<br><br>

    Executes a SQL query using the tables registered in this `BodoSQLContext`. This function should
    be used inside a `@bodo.jit` function.

    ***Arguments***

    - `query`: The SQL query to execute. This function generates code that is compiled so the `query` argument is required
    to be a compile time constant.

   -  `params_dict`: A dictionary that maps a SQL usable name to Python variables. For more information please
    refer to [the BodoSQL named parameters section][bodosql_named_params].

    ***Returns***

    A `DataFrame` that results from executing the query.


- `bodosql.BodoSQLContext.add_or_replace_view(self, name: str, table: Union[pandas.DataFrame, TablePath])`
<br><br>

    Create a new `BodoSQLContext` from an existing `BodoSQLContext` by adding or replacing a table.

    ***Arguments***

    - `name`: The name of the table to add. If the name already exists references to that table
    are removed from the new context.

    - `table`: The table object to add. `table` must be a `DataFrame` or `TablePath` object.

    ***Returns***

    A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` and inserts the new table specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.add_or_replace_view("t1", table)`



- `bodosql.BodoSQLContext.remove_view(self, name: str)`
<br><br>

    Creates a new `BodoSQLContext` from an existing context by removing the table with the
    given name. If the name does not exist, a `BodoError` is thrown.

    ***Arguments***

    - `name`: The name of the table to remove.

    ***Returns***

    A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` minus the table specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.remove_view("t1")`


- `bodosql.BodoSQLContext.add_or_replace_catalog(self, catalog: DatabaseCatalog)`
<br><br>

    Create a new `BodoSQLContext` from an existing context by replacing the `BodoSQLContext` object's `DatabaseCatalog` with
    a new catalog.

    ***Arguments***

    - `catalog`: The catalog to insert.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but replaces the old catalog with the new catalog specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.add_or_replace_catalog(catalog)`


- `bodosql.BodoSQLContext.remove_catalog(self)`
<br><br>

    Create a new `BodoSQLContext` from an existing context by removing its `DatabaseCatalog`.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but removes the old catalog.

    !!!note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.remove_catalog()`

## TablePath API

The `TablePath` API is a general purpose IO interface to specify IO sources. This API is meant
as an alternative to natively loading tables in Python inside JIT functions.
The `TablePath` API stores the user-defined data location and the storage type to load a table of interest.
For example, here is some sample code that loads two DataFrames from parquet using the `TablePath` API.

```py
bc = bodosql.BodoSQLContext(
    {
        "T1": bodosql.TablePath("my_file_path1.pq", "parquet"),
        "T2": bodosql.TablePath("my_file_path2.pq", "parquet"),
    }
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```

Here, the `TablePath` constructor doesn't load any data. Instead, a `BodoSQLContext` internally generates code to load the tables of interest after parsing the SQL query. Note that a `BodoSQLContext` loads all used tables from I/O *on every query*, which means that if users would like to perform multiple queries on the same data, they should consider loading the DataFrames once in a separate JIT function.

### API Reference

- `bodosql.TablePath(file_path: str, file_type: str, *, conn_str: Optional[str] = None, reorder_io: Optional[bool] = None)`
<br><br>

    Specifies how a DataFrame should be loaded from IO by a BodoSQL query. This
    can only load data when used with a `BodoSQLContext` constructor.

    ***Arguments***

    - `file_path`: Path to IO file or name of the table for SQL. This must constant at compile time if used inside JIT.

    - `file_type`: Type of file to load as a string. Supported values are ``"parquet"`` and ``"sql"``. This must constant at compile time if used inside JIT.

    - `conn_str`: Connection string used to connect to a SQL DataBase, equivalent to the conn argument to `pandas.read_sql`. This must be constant at compile time if used inside JIT and must be None if not loading from a SQL DataBase.

   - `reorder_io`: Boolean flag determining when to load IO. If `False`, all used tables are loaded before executing any of the query. If `True`, tables are loaded just before first use inside the query, which often results in decreased
    peak memory usage as each table is partially processed before loading the next table. The default value, `None`, behaves like `True`, but this may change in the future. This must be constant at compile time if used inside JIT.

## Database Catalogs

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

### SnowflakeCatalog

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

#### API Reference

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

### FileSystemCatalog

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
you must following the BodoSQL rules for [identifier case sensitivity](#identifier_case_sensitivity).

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

#### API Reference

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

## <a name="identifier_case_sensitivity"></a> Identifier Case Sensitivity

In BodoSQL all identifiers not wrapped in quotes are automatically converted to upper case.
If you are a Snowflake user who is using either the Snowflake Catalog or Table Path API, then this should not impact you and the
rules will be the same as Snowflake (i.e. identifiers are case-insensitive unless wrapped in quotes during table creation).
[See here for the Snowflake documentation.](https://docs.snowflake.com/en/sql-reference/identifiers-syntax#label-identifier-casing).

This means that the following queries are equivalent:

```sql
SELECT A FROM table1
```

```sql
SELECT a FROM TABLE1
```

When providing column or table names, identifiers will only match if the original name is in uppercase
For example, the following code will fail to compile because there is no match for TABLE1:

```py
@bodo.jit
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    return bc.sql("SELECT A FROM table1")
```

To match non-uppercase names you can use quotes to specify the name exactly as it appears in the BodoSQLContext
definition or the columns of a DataFrame. For example:

```py
@bodo.jit
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    return bc.sql("SELECT A FROM \"table1\"")
```

Similarly if you want an alias to be case sensitive then you will also need it to be wrapped in quotes:

```sql
SELECT A as "myIdentifier" FROM table1
```

If you provide DataFrames directly from Python or are using the TablePath API to load Parquet files, then please be advised
that the column names will be required to match exactly and for ease of use we highly recommend using uppercase column names.

## Performance Considerations

### Snowflake Views

Users may define views within their Snowflake account to enable greater query reuse.
Views may constitute performance bottlenecks because if a view is evaluated in Snowflake
Bodo will need to wait for the result before it can fetch data and may have less access
to optimizations.

To improve performance in these circumstances Bodo will attempt to expand any views into
the body of the query to allow Bodo to operate on the underlying tables. When this occurs
users should face no performance penalty for using views in their queries. However there are
a few situations in which this is not possible, namely

  * The Snowflake User passed to Bodo does not have permissions to determine the view definition.
  * The Snowflake User passed to Bodo does not have permissions to
    read all of the underlying tables.
  * The view is a materialized or secure view.

If for any reason Bodo is unable to expand the view, then the query will execute treating
the view as a table and delegate it to Snowflake.
