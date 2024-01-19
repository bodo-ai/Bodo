
Bodo 2024.1 Release (Date: 01/05/2024) {#January_2024}
========================================

## New Features and Improvements


### New Features:

- Distributed complex, two dimensional fft and `fftshift`.
- Added support for the function `TO_OBJECT`.
- Added support for `OUTER=>true` when calling `FLATTEN`.
- Support for creating a table with the `GENERATOR` function using only the `ROWCOUNT` argument.
- Increased support for variant arguments to numeric/datetime/string/array/object functions.
- Added support for `GET_PATH` and the alternate syntax using `:`.
- Support ARRAY_AGG on all types.
- Setting the environment variable BODO_DISABLE_SF_RESULT_CACHE to 1 will ensure that Snowflake doesn’t return results from its result cache when Bodo reads tables from Snowflake. This can be useful for performance testing but is not recommended for production use.


### Performance Improvements:


- General improvements to disk spilling performance.
- General improvements to the spilling functionality for better memory utilization in Join and Groupby.
- Enabled several plan optimizations such as simplifying null predicates and push predicates deeper into the plan.


### Bug Fixes:


- Coalesce type coercion now matches snowflake behavior
- `OBJECT_AGG` now omits any rows where the key or value is null in the final object.
- Fixed a bug where default Snowflake connection settings could cause a timeout for long running writes. We now set ABORT_DETACHED_QUERY=False automatically to avoid this issue.


### Usability Improvements


- Improved the specificity of error messages regarding Snowflake UDFs to help better explain limitations in Bodo functionality for UDFs. Future releases will expand this functionality.
- Using an aggregation function other than first on a semi-structured column now throws an error.


## 2024.1.2 New Features and Improvements


### New Features:


- Support for using semi-structured columns as keys to `#!sql JOIN` and `#!sql GROUP BY`.
- Support for `#!sql APPROX_PERCENTILE` as a window function.
- Support for object literals as syntactic sugar for `#!sql OBJECT_CONSTRUCT`.
- Added limited support for `#!sql PARSE_JSON` when it can be re-written as a call to JSON_EXTRACT_PATH_TEXT followed by a cast to another type. E.g.: `#!sql PARSE_JSON(S):field::integer` can be rewritten as `#!sql JSON_EXTRACT_PATH_TEXT(S, ‘field’)::integer`.
- Support for sub-millisecond resolution in `#!sql TIME` and `#!sql TIMESTAMP` literals.
- Support `#!sql SELECT * EXCLUDING (columns)` syntax.
- We have updated the BodoSQL identifier handling to be more consistent with Snowflake and the SQL standard. This has the following new behavior which matches Snowflake:
   
    * Unquoted identifiers are converted to uppercase letters.
    * Quoted identifiers retain their current casing.
    * Identifier matching is now case sensitive after this conversion. Please refer to our documentation for more detailed examples.

- Support boolean inputs in `#!sql TO_NUMBER.`
- Improved support for nested datatypes in BodoSQL originating from Python.
- Added support for Object literals.
- Support for distributed transpose of Numpy arrays.
- Distributed support for real/imaginary components of complex Numpy arrays.
- Support for distributed multi-dimensional getitem for Numpy arrays (only scalar output currently).
- Remote spilling to S3 is now enabled by default (on AWS workspaces) for increased reliability.


### Bug Fixes:


- Fixed `#!sql OBJECT_DELETE`, `#!sql OBJECT_PICK` and `#!sql OBJECT_CONSTRUCT` behavior when used inside of a CASE statement.



## 2024.1.3 New Features and Improvements


### New Features:


- Support casting arrays to null array.


### Performance Improvements:


- Improved literal cast constant folding, which is especially impactful for datetime literals.
- General planner improvements on filter inference/simplification.
- Improved performance of `np.interp`.



### Bug Fixes:


- Ensure datetime literal default precision is always 9.


## 2024.1.4 New Features and Improvements


### New Features:


- Expanded support for conditions in non-inner joins to always support conditions that only apply to one side of the join.
- Expanded support of conditional functions with variant inputs, and for accessing array indices & object fields when the array or object is a variant.
- Support for the `#!sql UUID_STRING` function.
- Support string literals enclosed by dollar signs, e.g. `#!sql $$Hello World$$`.
- Support `#!sql OBJECT_AGG` as a window function.
- Support for comparing decimal columns to integer and float columns.
- Support for semi-structured arrays in `groupby.apply`.
- Double quoted unit arguments to `#!sql  DATE_TRUNC` are now supported.
- Support for some Snowflake UDFs when the function body is an expression or a query with no arguments. Future releases will continue to expand Snowflake UDF support. 


### Bug Fixes:


- Fixed behavior of `#!sql LISTAGG` when all data is null to match Snowflake.
