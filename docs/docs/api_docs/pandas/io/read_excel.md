# pd.read_excel

`pandas.read_excel`

- output dataframe cannot be parallelized automatically yet.
- only arguments `io`, `sheet_name`, `header`, `names`, `comment`,
  `dtype`, `skiprows`, `parse_dates` are supported.
- `io` should be a string and is required.
- Either `names` and `dtype` arguments should be provided to
  enable type inference, or `io` should be inferrable as a
  constant string. This is required so bodo can infer the types at
  compile time, see [compile time constants][require_constants]
- `sheet_name`, `header`, `comment`, and `skiprows` should be
  constant if provided.
- `names` and `parse_dates` should be constant lists if provided.
- `dtype` should be a constant dictionary of strings and types if
  provided.
