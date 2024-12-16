# DataFrame

Bodo provides extensive DataFrame support. This section covers the DataFrame API.

## Creation

| Function                      | Description        |
|-------------------------------|--------------------|
| [`pd.DataFrame`][pddataframe] | Create a DataFrame |

## Attributes and underlying data

| Function                                                 | Description                                                                  |
|----------------------------------------------------------|------------------------------------------------------------------------------|
| [`pd.DataFrame.columns`][pddataframecolumns]             | The column labels of the DataFrame                                           |
| [`pd.DataFrame.dtypes`][pddataframedtypes]               | Return the dtypes in the DataFrame                                           |
| [`pd.DataFrame.empty`][pddataframeempty]                 | Indicator whether DataFrame is empty                                         |
| [`pd.DataFrame.index`][pddataframeindex]                 | The index (row labels) of the DataFrame                                      |
| [`pd.DataFrame.ndim`][pddataframendim]                   | Number of axes / array dimensions                                            |
| [`pd.DataFrame.select_dtypes`][pddataframeselect_dtypes] | Return a subset of the DataFrame's columns based on the column dtypes        |
| [`pd.DataFrame.filter`][pddataframefilter]               | Subset the DataFrame rows or columns according to the specified index labels |
| [`pd.DataFrame.shape`][pddataframeshape]                 | Return a tuple representing the dimensionality of the DataFrame              |
| [`pd.DataFrame.size`][pddataframesize]                   | Number of elements in the DataFrame                                          |
| [`pd.DataFrame.to_numpy`][pddataframeto_numpy]           | Return a Numpy representation of the DataFrame                               |
| [`pd.DataFrame.values`][pddataframevalues]               | Return a Numpy representation of the DataFrame                               |

## Conversion

| Function                                                 | Description                                       |
|----------------------------------------------------------|---------------------------------------------------|
| [`pd.DataFrame.astype`][pddataframeastype]               | Cast a pandas object to a specified dtype         |
| [`pd.DataFrame.copy`][pddataframecopy]                   | Make a copy of the DataFrame                      |
| [`pd.DataFrame.isna`][pddataframeisna]                   | Detect missing values                             |
| [`pd.DataFrame.isnull`][pddataframeisnull]               | Detect missing values                             |
| [`pd.DataFrame.notna`][pddataframenotna]                 | Detect existing (non-missing) values              |
| [`pd.DataFrame.notnull`][pddataframenotnull]             | Detect existing (non-missing) values              |
| [`pd.DataFrame.info`][pddataframeinfo]                   | Print a concise summary of a DataFrame            |
| [`pd.DataFrame.infer_objects`][pddataframeinfer_objects] | Attempt to infer better dtypes for object columns |

## Indexing, iteration

| Function                                           | Description                                                      |
|----------------------------------------------------|------------------------------------------------------------------|
| [`pd.DataFrame.head`][pddataframehead]             | Return the first `n` rows                                        |
| [`pd.DataFrame.iat`][pddataframeiat]               | Access a single value for a row/column pair by integer position  |
| [`pd.DataFrame.iloc`][pddataframeiloc]             | Purely integer-location based indexing for selection by position |
| [`pd.DataFrame.insert`][pddataframeinsert]         | Insert column into DataFrame at specified location               |
| [`pd.DataFrame.isin`][pddataframeisin]             | Determine if values are contained in a Series or DataFrame       |
| [`pd.DataFrame.itertuples`][pddataframeitertuples] | Iterate over DataFrame rows as namedtuples                       |
| [`pd.DataFrame.query`][pddataframequery]           | Query the columns of a DataFrame with a boolean expression       |
| [`pd.DataFrame.tail`][pddataframetail]             | Return the last `n` rows                                         |
| [`pd.DataFrame.where`][pddataframewhere]           | Replace values where the condition is False                      |
| [`pd.DataFrame.mask`][pddataframemask]             | Replace values where the condition is True                       |

## Function Application, GroupBy & Window

| Function                                     | Description                                              |
|----------------------------------------------|----------------------------------------------------------|
| [`pd.DataFrame.apply`][pddataframeapply]     | Apply a function along an axis of the DataFrame          |
| [`pd.DataFrame.groupby`][pddataframegroupby]  | Group DataFrame using a mapper or by a Series of columns |
| [`pd.DataFrame.rolling`][pddataframerolling] | Provide rolling window calculations                      |

## Computations / Descriptive Stats

| Function                                               | Description                                                       |
|--------------------------------------------------------|-------------------------------------------------------------------|
| [`pd.DataFrame.abs`][pddataframeabs]                   | Return a DataFrame with absolute numeric value of each element    |
| [`pd.DataFrame.corr`][pddataframecorr]                 | Compute pairwise correlation of columns, excluding NA/null values |
| [`pd.DataFrame.count`][pddataframecount]               | Count non-NA cells for each column or row                         |
| [`pd.DataFrame.cov`][pddataframecov]                   | Compute pairwise covariance of columns, excluding NA/null values  |
| [`pd.DataFrame.cumprod`][pddataframecumprod]           | Return cumulative product over a DataFrame or Series axis         |
| [`pd.DataFrame.cumsum`][pddataframecumsum]             | Return cumulative sum over a DataFrame or Series axis             |
| [`pd.DataFrame.describe`][pddataframedescribe]         | Generate descriptive statistics                                   |
| [`pd.DataFrame.diff`][pddataframediff]                 | First discrete difference of element                              |
| [`pd.DataFrame.max`][pddataframemax]                   | Return the maximum of the values for the requested axis           |
| [`pd.DataFrame.mean`][pddataframemean]                 | Return the mean of the values for the requested axis              |
| [`pd.DataFrame.median`][pddataframemedian]             | Return the median of the values for the requested axis            |
| [`pd.DataFrame.min`][pddataframemin]                   | Return the minimum of the values for the requested axis           |
| [`pd.DataFrame.nunique`][pddataframenunique]           | Count distinct observations over requested axis                   |
| [`pd.DataFrame.pct_change`][pddataframepct_change]     | Percentage change between the current and a prior element         |
| [`pd.DataFrame.pipe`][pddataframepipe]                 | Apply func(self, *args, **kwargs)                                 |
| [`pd.DataFrame.prod`][pddataframeprod]                 | Return the product of the values for the requested axis           |
| [`pd.DataFrame.product`][pddataframeproduct]           | Return the product of the values for the requested axis           |
| [`pd.DataFrame.quantile`][pddataframequantile]         | Return values at the given quantile over requested axis           |
| [`pd.DataFrame.rank`][pddataframerank]                 | Compute numerical data ranks (1 through n) along axis             |
| [`pd.DataFrame.std`][pddataframestd]                   | Return sample standard deviation over requested axis              |
| [`pd.DataFrame.sum`][pddataframesum]                   | Return the sum of the values for the requested axis               |
| [`pd.DataFrame.var`][pddataframevar]                   | Return unbiased variance over requested axis                      |
| [`pd.DataFrame.memory_usage`][pddataframememory_usage] | Return the memory usage of each column in bytes                   |

## Reindexing / Selection / Label manipulation

| Function                                                     | Description                                                       |
|--------------------------------------------------------------|-------------------------------------------------------------------|
| [`pd.DataFrame.drop`][pddataframedrop]                       | Drop specified labels from rows or columns                        |
| [`pd.DataFrame.drop_duplicates`][pddataframedrop_duplicates] | Return DataFrame with duplicate rows removed                      |
| [`pd.DataFrame.duplicated`][pddataframeduplicated]           | Return boolean Series denoting duplicate rows                     |
| [`pd.DataFrame.first`][pddataframefirst]                     | Select initial periods of time series data based on a date offset |
| [`pd.DataFrame.idxmax`][pddataframeidxmax]                   | Return the row label of the maximum value                         |
| [`pd.DataFrame.idxmin`][pddataframeidxmin]                   | Return the row label of the minimum value                         |
| [`pd.DataFrame.last`][pddataframelast]                       | Select final periods of time series data based on a date offset   |
| [`pd.DataFrame.rename`][pddataframerename]                   | Alter axes labels                                                 |
| [`pd.DataFrame.reset_index`][pddataframereset_index]         | Reset the index of the DataFrame                                  |
| [`pd.DataFrame.set_index`][pddataframeset_index]             | Set the DataFrame index using existing columns                    |
| [`pd.DataFrame.take`][pddataframetake]                       | Return the elements in the given positional indices along an axis |

## Missing data handling

| Function                                     | Description                                   |
|----------------------------------------------|-----------------------------------------------|
| [`pd.DataFrame.dropna`][pddataframedropna]   | Remove missing values                         |
| [`pd.DataFrame.fillna`][pddataframefillna]   | Fill NA/NaN values using the specified method |
| [`pd.DataFrame.replace`][pddataframereplace] | Replace values given in to_replace with value |

  
## Reshaping, sorting, transposing

| Function                                             | Description                                                              |
|------------------------------------------------------|--------------------------------------------------------------------------|
| [`pd.DataFrame.explode`][pddataframeexplode]         | Transform each element of a list-like to a row, replicating index values |
| [`pd.DataFrame.melt`][pddataframemelt]               | Unpivot a DataFrame from wide to long format                             |
| [`pd.DataFrame.pivot`][pddataframepivot]             | Return reshaped DataFrame organized by given index / column values       |
| [`pd.DataFrame.pivot_table`][pddataframepivot_table] | Create a spreadsheet-style pivot table as a DataFrame                    |
| [`pd.DataFrame.sample`][pddataframesample]           | Return a random sample of items from an axis of object                   |
| [`pd.DataFrame.sort_index`][pddataframesort_index]   | Sort object by labels (along an axis)                                    |
| [`pd.DataFrame.sort_values`][pddataframesort_values] | Sort by the values along either axis                                     |
| [`pd.DataFrame.to_string`][pddataframeto_string]     | Render a DataFrame to a console-friendly tabular output                  |

## Combining / joining / merging

| Function                                   | Description                                                          |
|--------------------------------------------|----------------------------------------------------------------------|
| [`pd.DataFrame.append`][pddataframeappend] | Append rows of other to the end of caller, returning a new object    |
| [`pd.DataFrame.assign`][pddataframeassign] | Assign new columns to a DataFrame                                    |
| [`pd.DataFrame.join`][pddataframejoin]     | Join columns with other DataFrame either on index or on a key column |
| [`pd.DataFrame.merge`][pddataframemerge]   | Merge DataFrame or named Series objects with a database-style join   |

## Time series-related

| Function                                | Description                                                         |
|-----------------------------------------|---------------------------------------------------------------------|
| [`pd.DataFrame.shift`][pddataframeshift] | Shift index by desired number of periods with an optional time freq |

## Serialization, IO, Conversion

| Function                                           | Description                                           |
|----------------------------------------------------|-------------------------------------------------------|
| [`pd.DataFrame.to_csv`][pddataframeto_csv]         | Write object to a comma-separated values (csv) file   |
| [`pd.DataFrame.to_json`][pddataframeto_json]       | Convert the object to a JSON string                   |
| [`pd.DataFrame.to_parquet`][pddataframeto_parquet] | Write a DataFrame to the binary parquet format        |
| [`pd.DataFrame.to_sql`][pddataframeto_sql]         | Write records stored in a DataFrame to a SQL database |

## Plotting

| Function                               | Description |
|----------------------------------------|-------------|
| [`pd.DataFrame.plot`][pddataframeplot] | Plot data   |
