# General Functions

General functions are the most commonly used functions in Pandas. They include functions for data manipulation, data cleaning, data merging, and more.


## Data Manipulations

| Function                        | Description                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------|
| [pd.concat][pdconcat]           | Concatenate pandas objects along a particular axis with optional set logic along the other axes. |
| [pd.crosstab][pdcrosstab]       | Compute a simple cross-tabulation of two (or more) factors.                                      |
| [pd.cut][pdcut]                 | Bin values into discrete intervals.                                                              |
| [pd.qcut][pdqcut]               | Quantile-based discretization function.                                                          |
| [pd.get_dummies][pdget_dummies] | Convert categorical variable into dummy/indicator variables.                                     |
| [pd.merge][pdmerge]             | Merge DataFrame or named Series objects with a database-style join.                              |
| [pd.pivot][pdpivot]             | Reshape data (produce a “pivot” table) based on column values.                                   |
| [pd.pivot_table][pdpivot_table] | Create a spreadsheet-style pivot table as a DataFrame.                                           |
| [pd.unique][pdunique]           | Hash table-based unique.                                                                         |

## Top Level Missing Data

| Function                | Description                           |
|-------------------------|---------------------------------------|
| [pd.isna][pdisna]       | Detect missing values.                |
| [pd.notna][pdnotna]     | Detect existing (non-missing) values. |
| [pd.isnull][pdisnull]   | Detect missing values.                |
| [pd.notnull][pdnotnull] | Detect existing (non-missing) values. |


## Top Level Conversions

| Function                      | Description                         |
|-------------------------------|-------------------------------------|
| [pd.to_numeric][pdto_numeric] | Convert argument to a numeric type. |
