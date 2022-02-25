# Input/Output {#pandas-f-in}

See more in [File IO][file_io], such as
[S3][] and [HDFS][] configuration requirements.

#### pd.read_csv


- <code><apihead>pandas.<apiname>read_csv</apiname></apihead></code>
<br><br>
    -   [example usage and more system specific instructions][csv-section]
    -   `filepath_or_buffer` should be a string and is required. It
        could be pointing to a single CSV file, or a directory
        containing multiple partitioned CSV files (must have `csv` file
        extension inside directory).
    -   Arguments `sep`, `delimiter`, `header`, `names`, `index_col`,
        `usecols`, `dtype`, `nrows`, `skiprows`, `chunksize`,
        `parse_dates`, and `low_memory` are supported.
    -   Either `names` and `dtype` arguments should be provided to
        enable type inference, or `filepath_or_buffer` should be
        inferrable as a constant string. This is required so bodo can
        infer the types at compile time, see [compile time constants][require_constants]
    -   `names`, `usecols`, `parse_dates` should be constant lists.
    -   `dtype` should be a constant dictionary of strings and types.
    -   `skiprows` must be an integer or list of integers and if it is
        not a constant, `names` must be provided to enable type
        inference.
    -   `chunksize` is supported for uncompressed files only.
    -   `low_memory` internally process file in chunks while parsing. In
        Bodo this is set to `False` by default.
    -   When set to `True`, Bodo parses file in chunks but
        like Pandas the entire file is read into a single DataFrame
        regardless.
    -   If you want to load data in chunks, use the `chunksize`
        argument.
    -   When a CSV file is read in parallel (distributed mode) and each
        process reads only a portion of the file, reading columns that
        contain line breaks is not supported.

#### pd.read_excel


- <code><apihead>pandas.<apiname>read_excel</apiname></apihead></code>
<br><br>
    -   output dataframe cannot be parallelized automatically yet.
    -   only arguments `io`, `sheet_name`, `header`, `names`, `comment`,
        `dtype`, `skiprows`, `parse_dates` are supported.
    -   `io` should be a string and is required.
    -   Either `names` and `dtype` arguments should be provided to
        enable type inference, or `io` should be inferrable as a
        constant string. This is required so bodo can infer the types at
        compile time, see [compile time constants][require_constants]
    -   `sheet_name`, `header`, `comment`, and `skiprows` should be
        constant if provided.
    -   `names` and `parse_dates` should be constant lists if provided.
    -   `dtype` should be a constant dictionary of strings and types if
        provided.

#### pd.read_sql


- <code><apihead>pandas.<apiname>read_sql</apiname></apihead></code>
<br><br>
    -   [example usage and more system specific instructions][sql-section]
    -   Argument `sql` is supported but only as a string form.
        SQLalchemy `Selectable` is not supported. There is
        no restriction on the form of the sql request.
    -   Argument `con` is supported but only as a string form.
        SQLalchemy `connectable` is not supported.
    -   Argument `index_col` is supported.
    -   Arguments `chunksize`, `column`, `coerce_float`, `params` are
        not supported.

#### pd.read_parquet


- <code><apihead>pandas.<apiname>read_parquet</apiname></apihead></code>
<br><br>
    -   [example usage and more system specific instructions][parquet-section]
    -   Arguments `path` and `columns` are supported. `columns` should
        be a constant list of strings if provided.
    -   Argument `anon` of `storage_options` is supported for S3
        filepaths.
    -   If `path` can be inferred as a constant (e.g. it is a function
        argument), Bodo finds the schema from file at compilation time.
        Otherwise, schema should be provided using the [numba syntax](https://numba.pydata.org/numba-doc/latest/reference/types.html){target=blank}.
        
        For example:
        ```py
        @bodo.jit(locals={'df':{'A': bodo.float64[:],
                                'B': bodo.string_array_type}})
        def impl(f):
          df = pd.read_parquet(f)
          return df
        ```
        
#### `pd.read_json`


- <code><apihead>pandas.<apiname>read_json</apiname></apihead></code>
<br><br>
    -   [Example usage and more system specific instructions][json-section]
    -   Only supports reading [JSON Lines text file format](http://jsonlines.org/){target=blank}
        (`pd.read_json(filepath_or_buffer, orient='records', lines=True)`)
        and regular multi-line JSON
        file(`pd.read_json(filepath_or_buffer, orient='records', lines=False)`).
    -   Argument `filepath_or_buffer` is supported: it can point to a
        single JSON file, or a directory containing multiple partitioned
        JSON files. When reading a directory, the JSON files inside the
        directory must be [JSON Lines text file
        format](http://jsonlines.org/){target=blank} with `json` file extension.
    -   Argument `orient = 'records'` is used as default, instead of
        Pandas' default `'columns'` for dataframes. `'records'` is the
        only supported value for `orient`.
    -   Argument `typ` is supported. `'frame'` is the only supported
        value for `typ`.
    -   `filepath_or_buffer` must be inferrable as a constant string.
        This is required so bodo can infer the types at compile time,
        see [compile time constants][require_constants].
    -   Arguments `convert_dates`, `precise_float`, `lines` are
        supported.


