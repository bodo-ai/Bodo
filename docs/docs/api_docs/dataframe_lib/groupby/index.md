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
- [`DataFrameGroupBy.kurtosis`][frame-agg-kurtosis]
- [`DataFrameGroupBy.kurt`][frame-agg-kurtosis]
- [`DataFrameGroupBy.nunique`][frame-agg-nunique]
- [`DataFrameGroupBy.size`][frame-agg-size]
- [`DataFrameGroupBy.first`][frame-agg-first]
- [`DataFrameGroupBy.last`][frame-agg-last]
- [`DataFrameGroupBy.any`][frame-agg-any]
- [`DataFrameGroupBy.all`][frame-agg-all]

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
- [`SeriesGroupBy.kurtosis`][series-agg-kurtosis]
- [`SeriesGroupBy.kurt`][series-agg-kurtosis]
- [`SeriesGroupBy.nunique`][series-agg-nunique]
- [`SeriesGroupBy.size`][series-agg-size]
- [`SeriesGroupBy.first`][series-agg-first]
- [`SeriesGroupBy.last`][series-agg-last]
- [`SeriesGroupBy.any`][series-agg-any]
- [`SeriesGroupBy.all`][series-agg-all]


[frame-agg-sum]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.sum.html
[frame-agg-count]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.count.html
[frame-agg-min]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.min.html
[frame-agg-max]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.max.html
[frame-agg-median]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.median.html
[frame-agg-mean]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.mean.html
[frame-agg-std]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.std.html
[frame-agg-var]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.var.html
[frame-agg-skew]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.skew.html
[frame-agg-kurtosis]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.kurt.html
[frame-agg-nunique]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.nunique.html
[frame-agg-size]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.size.html
[frame-agg-first]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.first.html
[frame-agg-last]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.last.html
[frame-agg-any]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.any.html
[frame-agg-all]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.all.html

[series-agg-sum]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.sum.html
[series-agg-count]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.count.html
[series-agg-min]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.min.html
[series-agg-max]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.max.html
[series-agg-median]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.median.html
[series-agg-mean]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.mean.html
[series-agg-std]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.std.html
[series-agg-var]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.var.html
[series-agg-skew]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.skew.html
[series-agg-kurtosis]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.kurt.html
[series-agg-nunique]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.nunique.html
[series-agg-size]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.size.html
[series-agg-first]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.first.html
[series-agg-last]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.last.html
[series-agg-any]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.any.html
[series-agg-all]: https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.all.html
