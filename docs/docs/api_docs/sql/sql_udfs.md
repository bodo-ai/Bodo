# User Defined Functions (UDFs) and User Defined Table Functions (UDTFs)

BodoSQL supports using Snowflake UDFs and UDTFs in queries and views. To make
UDFs and UDTFs available in BodoSQL, you must first register and define them inside
your Snowflake account using the appropriate `#!sql create function` command. Once
the function is created, so long as your user can access the
[function's metadata](https://docs.snowflake.com/en/sql-reference/info-schema/functions),
BodoSQL can process queries that use the function.

## Usage

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

## Calling Convention Best Practices

When calling either a UDF or a UDTF, we strongly recommend always using the named calling
convention. This is because UDFs support overloaded definitions and using distinct names
is the safest way to ensure you are calling the correct function. For more information see
this section of the [Snowflake Documentation](https://docs.snowflake.com/en/developer-guide/udf-stored-procedure-naming-conventions#overloading-procedures-and-functions).
Even if you are not currently using an overloaded function, we encourage this practice in case
the function is overloaded in the future.

## Requirements

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

## Performance

BodoSQL supports UDFs and UDTFs by inlining the function body directly into the
body of the query. This means that users of these functions should achieve the same
performance as if they had written the function body directly into the query.

For complex UDFs or UDTFs, naively executing the function body may require producing a correlated
subquery, an operation in which a query must be executed once per row in another table.
This can cause a significant performance hit, so BodoSQL undergoes a process called
decorrelation to rewrite the query in terms of much more efficient joins. If BodoSQL
is not able to rewrite a query, then it will raise an error indicating a correlation
could not be fully removed.

## Overloaded Definition Priority

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
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|
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
