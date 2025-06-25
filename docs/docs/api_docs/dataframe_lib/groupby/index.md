# GroupBy
The DataFrame Library supports grouping BodoDataFrames on columns and aggregating the grouped data via the `bodo.pandas.DataFrameGroupBy` and `bodo.pandas.SeriesGroupBy` classes. An instance of one of these classes will be returned when calling the groupby method (`BodoDataFrame.groupby`).

!!! note
	Currently the DataFrame Library only supports a subset of aggregation functions with their default parameters (listed below). Future releases will add more functionality including transformation and filtering. For now, if an unsupported method or property of a `bodo.pandas.SeriesGroupBy` or `bodo.pandas.DataFrameGroupBy` is encountered, the user's code will gracefully fall back to Pandas.

## Function Application

## `DataFrameGroupby` Computations / descriptive stats

## `SeriesGroupby` Computations / descriptive stats
