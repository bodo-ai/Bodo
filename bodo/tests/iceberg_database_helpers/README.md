# Generating test datasets

This directory contains scripts to generate datasets with various characteristics
that we would want to test.
In particular, we have the following datasets:

- Simple Datasets: Very simple datasets, without any additional inserts, schema evolution, etc.

  - SIMPLE_NUMERIC_TABLE: Numeric datatypes
  - SIMPLE_STRING_TABLE: Strings
  - SIMPLE_BOOL_BINARY_TABLE: Boolean and Binary datatypes
  - SIMPLE_STRUCT_TABLE: Structs with numeric and string elements
  - SIMPLE_LIST_TABLE: Lists with numeric and string elements
  - SIMPLE_MAP_TABLE: Maps with various numeric types and strings as keys and values
  - SIMPLE_DT_TSZ_TABLE: date (w/o time or tz) and timestampz (w/ tz) columns.
    - time (time of day w/o date or tz) and timestamp (w/o tz) are not supported by Spark, so will need to be done later.

- Schema Evolution: Datasets that have gone through schema evolutions such as adding, removing, renaming, changing dtype of, moving positions of columns.

  - schema_evolved_table

- Subsets of Files: Datasets where we will be reading a subset of rows

  - file_subset_deleted_rows_table: Remove rows from a previous insert.
  - file_subset_empty_files_table: Create files where current set of columns don't exist, so we'll read NaNs from them.
  - file_subset_partial_file: Delete some rows from a file.
    - NOTE: This doesn't work as expected since instead of storing information that a subset of rows should be used, Spark writes a whole new file.

- Partitioning: Add/Remove/Modify partitions on datasets to test that we handle these transformations correctly.

  - partitions_general_table: Do the above specified operations on numeric and string dtypes.
  - partitions_dt_table: Include a date column and evolve partitioning on this (year, month, day)
  - partitions_dropped_dt_table: Similar to partitions_dt_table except intermediate partitions are dropped rather than just updated, changing what can be determined from the partition spec.

- Filter-Pushdown: Change the partition field between commits to test that we handle these correctly.

  - filter_pushdown_test_table: Change partition fields (year --> month --> day) and eventually remove partitioning. Evolve schema (renaming column and moving its position) between these changes.

To generate any of these datasets, set up Iceberg and PySpark and run the corresponding script (`python <TABLE_NAME>.py`). The dataset
will be created at `./iceberg_db/<TABLE_NAME>`. This was tested with `pyspark==3.2` at the time of writing.

You can use `python create_all_tables.py` to generate all the test datasets.

NOTE: Spark may throw errors saying `java.io.FileNotFoundException: File iceberg_db/SIMPLE_NUMERIC_TABLE/metadata/version-hint.text does not exist`
which seems to be harmless.
