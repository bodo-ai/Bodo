# pd.read_csv

`pandas.read_csv`

- [example usage and more system specific instructions][csv-section]
  `filepath_or_buffer` should be a string and is required. It
  could be pointing to a single CSV file, or a directory
  containing multiple partitioned CSV files (must have `csv` file
  extension inside directory).

- Arguments `sep`, `delimiter`, `header`, `names`, `index_col`,
  `usecols`, `dtype`, `nrows`, `skiprows`, `chunksize`,
  `parse_dates`, and `low_memory` are supported.

- Argument `anon` of `storage_options` is supported for S3
  filepaths.

- Either `names` and `dtype` arguments should be provided to
  enable type inference, or `filepath_or_buffer` should be
  inferrable as a constant string. This is required so bodo can
  infer the types at compile time, see [compile time constants][require_constants]

- `names`, `usecols`, `parse_dates` should be constant lists.

- `dtype` should be a constant dictionary of strings and types.

- `skiprows` must be an integer or list of integers and if it is
  not a constant, `names` must be provided to enable type
  inference.

- `chunksize` is supported for uncompressed files only.

- `low_memory` internally process file in chunks while parsing. In
  Bodo this is set to `False` by default.

- When set to `True`, Bodo parses file in chunks but
  like Pandas the entire file is read into a single DataFrame
  regardless.

- If you want to load data in chunks, use the `chunksize`
  argument.

- When a CSV file is read in parallel (distributed mode) and each
  process reads only a portion of the file, reading columns that
  contain line breaks is not supported.

- `_bodo_read_as_dict` is a Bodo specific argument which forces
  the specified string columns to be read with dictionary-encoding.
  Dictionary-encoding stores data in memory in an efficient
  manner and is most effective when the column has many repeated values.
  Read more about dictionary-encoded layout
  [here](https://arrow.apache.org/docs/format/Columnar.html#dictionary-encoded-layout){target=blank}.

  For example:

  ```py
  @bodo.jit()
  def impl(f):
    df = pd.read_csv(f, _bodo_read_as_dict=["A", "B", "C"])
    return df
  ```
