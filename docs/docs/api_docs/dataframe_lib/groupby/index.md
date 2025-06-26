# GroupBy
The DataFrame Library supports grouping BodoDataFrames on columns and aggregating the grouped data via the `bodo.pandas.DataFrameGroupBy` and `bodo.pandas.SeriesGroupBy` classes. An instance of one of these classes will be returned when calling the groupby method (`BodoDataFrame.groupby`).

!!! note
	Currently the DataFrame Library only supports a subset of aggregation functions with their default parameters (listed below). Future releases will add more functionality including transformation and filtering. For now, if an unsupported method or property of a `bodo.pandas.SeriesGroupBy` or `bodo.pandas.DataFrameGroupBy` is encountered, the user's code will gracefully fall back to Pandas.

## Function Application

- [`DataFrameGroupBy.agg`][]
- [`SeriesGroupBy.agg`][]

## `DataFrameGroupBy` Computations / descriptive stats

- [`DataFrameGroupBy.sum`][]
- [`DataFrameGroupBy.count`][]
- [`DataFrameGroupBy.min`][]
- [`DataFrameGroupBy.max`][]
- [`DataFrameGroupBy.median`][]
- [`DataFrameGroupBy.mean`][]
- [`DataFrameGroupBy.std`][]
- [`DataFrameGroupBy.var`][]
- [`DataFrameGroupBy.skew`][]
- [`DataFrameGroupBy.nunique`][]
- [`DataFrameGroupBy.size`][]

## `SeriesGroupby` Computations / descriptive stats

- [`SeriesGroupBy.sum`][]
- [`SeriesGroupBy.count`][]
- [`SeriesGroupBy.min`][]
- [`SeriesGroupBy.max`][]
- [`SeriesGroupBy.median`][]
- [`SeriesGroupBy.mean`][]
- [`SeriesGroupBy.std`][]
- [`SeriesGroupBy.var`][]
- [`SeriesGroupBy.skew`][]
- [`SeriesGroupBy.nunique`][]
- [`SeriesGroupBy.size`][]
