# pd.read_parquet

`pandas.read_parquet`

-   [example usage and more system specific instructions][parquet-section]
-   Arguments `path` and `columns` are supported. `columns` should
    be a constant list of strings if provided.
    `path` can be a string or list. If string, must be a path to a file
    or a directory, or a glob string. If a list, must contain paths
    to parquet files (not directories) or glob strings.
-   Argument `anon` of `storage_options` is supported for S3
    filepaths.
-   If `path` can be inferred as a constant (e.g. it is a function
    argument), Bodo finds the schema from file at compilation time.
    Otherwise, schema should be provided using the [numba syntax](https://numba.pydata.org/numba-doc/latest/reference/types.html){target=blank}.
    
    For example:
    ```py
    @bodo.jit(locals={'df':{'A': bodo.types.float64[:],
                            'B': bodo.string_array_type}})
    def impl(f):
      df = pd.read_parquet(f)
      return df
    ```

-   `_bodo_input_file_name_col` is a Bodo specific argument.
    When specified, a column with this
    name is added to the dataframe consisting of the name of the file the
    row was read from. This is similar to SparkSQL's 
    [`input_file_name`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.input_file_name.html){target=blank} function.

    For example:
    ```py
    @bodo.jit
    def impl(f):
      df = pd.read_parquet(f, _bodo_input_file_name_col="fname")
      return df
    ```

-   `_bodo_read_as_dict` is a Bodo specific argument which forces 
    the specified string columns to be read with dictionary-encoding.
    Bodo automatically loads string columns using dictionary
    encoding when it determines it would be beneficial based on 
    a heuristic.
    Dictionary-encoding stores data in memory in an efficient
    manner and is most effective when the column has many repeated values.
    Read more about dictionary-encoded layout
    [here](https://arrow.apache.org/docs/format/Columnar.html#dictionary-encoded-layout){target=blank}.

    For example:
    ```py
    @bodo.jit
    def impl(f):
      df = pd.read_parquet(f, _bodo_read_as_dict=["A", "B", "C"])
      return df
    ```

        
