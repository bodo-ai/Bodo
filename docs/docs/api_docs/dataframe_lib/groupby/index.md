# GroupBy {#df-lib-groupby}

Bodo DataFrames supports grouping BodoDataFrames on columns
and aggregating the grouped data via the `bodo.pandas.DataFrameGroupBy` and `bodo.pandas.SeriesGroupBy` classes.
An instance of one of these classes will be returned when using the [`BodoDataFrame.groupby()`][frame-groupby] method.

!!! note
	Currently, Bodo DataFrames supports a subset of aggregation functions with the default parameters (listed below).
	Future releases will add more functionality including transformation and filtering.
	For now, if an unsupported method or property of a `bodo.pandas.SeriesGroupBy` or `bodo.pandas.DataFrameGroupBy` is encountered,
	the user's code [will gracefully fall back to Pandas][lazy-evaluation-and-fallback-to-pandas].

## Function Application

- [`DataFrameGroupBy.agg`][frame-agg]
- [`SeriesGroupBy.agg`][series-agg]
- [`DataFrameGroupby.apply][frame-apply]
- [`SeriesGroupBy.apply`][series-apply]

## DataFrameGroupBy Computations / Descriptive Stats

- [`DataFrameGroupBy.sum`][frame-agg-sum]
- [`DataFrameGroupBy.count`][frame-agg-count]
- [`DataFrameGroupBy.min`][frame-agg-min]
- [`DataFrameGroupBy.max`][frame-agg-max]
- [`DataFrameGroupBy.median`][frame-agg-median]
- [`DataFrameGroupBy.mean`][frame-agg-mean]
- [`DataFrameGroupBy.std`][frame-agg-std]
- [`DataFrameGroupBy.var`][frame-agg-var]
- [`DataFrameGroupBy.skew`][frame-agg-skew]
- [`DataFrameGroupBy.nunique`][frame-agg-nunique]
- [`DataFrameGroupBy.size`][frame-agg-size]
- [`DataFrameGroupBy.first`][frame-agg-first]
- [`DataFrameGroupBy.last`][frame-agg-last]

## SeriesGroupby Computations / Descriptive Stats

- [`SeriesGroupBy.sum`][series-agg-sum]
- [`SeriesGroupBy.count`][series-agg-count]
- [`SeriesGroupBy.min`][series-agg-min]
- [`SeriesGroupBy.max`][series-agg-max]
- [`SeriesGroupBy.median`][series-agg-median]
- [`SeriesGroupBy.mean`][series-agg-mean]
- [`SeriesGroupBy.std`][series-agg-std]
- [`SeriesGroupBy.var`][series-agg-var]
- [`SeriesGroupBy.skew`][series-agg-skew]
- [`SeriesGroupBy.nunique`][series-agg-nunique]
- [`SeriesGroupBy.size`][series-agg-size]
- [`SeriesGroupBy.first`][series-agg-first]
- [`SeriesGroupBy.last`][series-agg-last]


[frame-agg-sum]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sum.html
[frame-agg-count]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.count.html
[frame-agg-min]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.min.html
[frame-agg-max]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.max.html
[frame-agg-median]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.median.html
[frame-agg-mean]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.mean.html
[frame-agg-std]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.std.html
[frame-agg-var]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.var.html
[frame-agg-skew]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.skew.html
[frame-agg-nunique]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.nunique.html
[frame-agg-size]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.size.html
[frame-agg-first]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.first.html
[frame-agg-last]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.last.html

[series-agg-sum]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.sum.html
[series-agg-count]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.count.html
[series-agg-min]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.min.html
[series-agg-max]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.max.html
[series-agg-median]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.median.html
[series-agg-mean]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.mean.html
[series-agg-std]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.std.html
[series-agg-var]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.var.html
[series-agg-skew]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.skew.html
[series-agg-nunique]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.nunique.html
[series-agg-size]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.size.html
[series-agg-first]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.first.html
[series-agg-last]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.last.html
