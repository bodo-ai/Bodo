# DataFrame API
Bodo DataFrames supports Pandas DataFrame methods and accessors that are listed below. They can be accessed through `BodoDataFrame` and follow the same behavior as their Pandas equivalents. For details on usage, we link to either the corresponding Pandas documentation or relevant sections of the Bodo documentation, depending on the context.

!!! note
    If the user code encounters an unsupported Pandas API or an unsupported parameter, Bodo
	DataFrames gracefully falls back to native Pandas. See [overview][overview] of
	Bodo DataFrames for more info.


## Function application, GroupBy & window
- [`bodo.pandas.BodoDataFrame.apply`][bododfapply]
- [`bodo.pandas.BodoDataFrame.groupby`][bododfgroupby]
- [`bodo.pandas.BodoDataFrame.map_partitions`][bododfmappartitions]

---

## Reindexing / selection / label manipulation
- [`bodo.pandas.BodoDataFrame.head`][bododfhead]
- [`bodo.pandas.BodoDataFrame.reset_index`][bododfresetindex]
___

## Reshaping, sorting, transposing

- [`bodo.pandas.BodoDataFrame.sort_values`][bododfsortvalues]

___

## Serialization / IO / conversion

- [`bodo.pandas.BodoDataFrame.to_iceberg`][bododftoiceberg]
- [`bodo.pandas.BodoDataFrame.to_parquet`][bododftoparquet]
- [`bodo.pandas.BodoDataFrame.to_s3_vectors`][bododftos3vectors]



[bododfresetindex]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html

[bododfhead]: ../dataframe/head.md
[bododfapply]: ../dataframe/apply.md
[bododfmappartitions]: ../dataframe/map_partitions.md
[bododfsortvalues]: ../dataframe/sort_values.md
[bododftoiceberg]: ../dataframe/to_iceberg.md
[bododftoparquet]: ../dataframe/to_parquet.md
[bododftos3vectors]: ../dataframe/to_s3_vectors.md
[bododfgroupby]: ../dataframe/groupby.md