
Bodo 2024.1 Release (Date: 01/05/2024) {#January_2024}
========================================

### New Features:

- Distributed complex, two dimensional fft and fftshift
- Added support for the function `TO_OBJECT`.
- Added support for `OUTER=>true` when calling `FLATTEN`.
- Support for creating a table with the `GENERATOR` function using only the `ROWCOUNT` argument.
- Increased support for variant arguments to numeric/datetime/string/array/object functions.
- Added support for `GET_PATH` and the alternate syntax using `:`
- Support ARRAY_AGG on all types.
- Setting the environment variable BODO_DISABLE_SF_RESULT_CACHE to 1 will ensure that Snowflake doesn’t return results from its result cache when Bodo reads tables from Snowflake. This can be useful for performance testing but is not recommended for production use.


### Performance Improvements:

- General improvements to disk spilling performance
- General improvements to the spilling functionality for better memory utilization in Join and Groupby.
- Enabled several plan optimizations such as simplifying null predicates and push predicates deeper into the plan.

Bug Fixes:
- Coalesce type coercion now matches snowflake behavior
- `OBJECT_AGG` now omits any rows where the key or value is null in the final object.
- Fixed a bug where default Snowflake connection settings could cause a timeout for long running writes. We now set ABORT_DETACHED_QUERY=False automatically to avoid this issue.

### Usability Improvements

- Improved the specificity of error messages regarding Snowflake UDFs to help better explain limitations in Bodo functionality for UDFs. Future releases will expand this functionality.
- Using an aggregation function other than first on a semi-structured column now throws an error.
