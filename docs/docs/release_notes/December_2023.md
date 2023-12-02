
Bodo 2023.12 Release (Date: 12/01/2023) {#December_2023}
========================================

## New Features and Improvements


### New Features:


- Support for reading tables outside the default database.
- Initial support for expanding Snowflake views. When Bodo encounters a Snowflake view definition it will attempt to expand the view directly into the query, which will provide increased optimization opportunities and performance. Bodo will not be able to expand views if the user under which Bodo operates does not have permissions to access the underlying table or fetch the view definition. Bodo will not attempt to expand the view if it is a materialized view or a secure view, to comply with Snowflake users' expectations. If for any reason Bodo is unable to expand a view then the query will still execute by evaluating the view in Snowflake and reading it as a table. This is the first release offering this support, so the number of views that can be expanded will increase with future releases.
- Added support for the function `OBJECT_CONSTRUCT` including the syntactic sugar `OBJECT_CONSTRUCT(*)` so long as all of the values are of the same type (or trivially castable types).
- Add support for the functions `ARRAY_CONSTRUCT_COMPACT`, `ARRAY_REMOVE`, `ARRAY_REMOVE_AT`, `ARRAY_SLICE` and `TO_VARIANT`.
- Allow `FLATTEN` to be called on variant columns, so long as the variants contain array/json data.
- Bodo can now spill to S3.
- Support for GET function on array values


### Performance Improvements:


- Reduced peak memory usage when using the `FLATTEN` or `SPLIT_TO_TABLE` functions in streaming.


### Dependency Updates:


- Upgrade to Calcite 1.35