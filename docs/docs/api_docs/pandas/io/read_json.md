# `pd.read_json`

`pandas.read_json`

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
-   Argument `anon` of `storage_options` is supported for S3
    filepaths.

