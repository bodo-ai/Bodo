GroupBy {#pd_groupby_section}
=======

[//]: # (write an index page with links to all the groupby functions)

The `groupby` method is used to group data in a DataFrame or Series based on a given column or index level. The grouped data can then be aggregated, transformed, or filtered using various methods.

Bodo supports the following `groupby` methods:

| Function                                                                               | Description                                                               |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`pd.core.groupby.Groupby.agg`][pdcoregroupbygroupbyagg]                               | Compute aggregation using one or more operations over the specified axis. |
| [`pd.core.groupby.DataFrameGroupby.aggregate`][pdcoregroupbydataframegroupbyaggregate] | Compute aggregation using one or more operations over the specified axis. |
| [`pd.core.groupby.Groupby.apply`][pdcoregroupbygroupbyapply]                           | Apply a function to each group.                                           |
| [`pd.core.groupby.Groupby.count`][pdcoregroupbygroupbycount]                           | Count non-NA cells for each column or row.                                |
| [`pd.core.groupby.Groupby.cumsum`][pdcoregroupbygroupbycumsum]                         | Compute the cumulative sum of the values.                                 |
| [`pd.core.groupby.Groupby.first`][pdcoregroupbygroupbyfirst]                           | Compute the first value of the group.                                     |
| [`pd.core.groupby.Groupby.head`][pdcoregroupbygroupbyhead]                             | Return the first n rows of each group.                                    |
| [`pd.core.groupby.DataFrameGroupby.idxmax`][pdcoregroupbydataframegroupbyidxmax]       | Compute the index of the maximum value.                                   |
| [`pd.core.groupby.DataFrameGroupby.idxmin`][pdcoregroupbydataframegroupbyidxmin]       | Compute the index of the minimum value.                                   |
| [`pd.core.groupby.Groupby.last`][pdcoregroupbygroupbylast]                             | Compute the last value of the group.                                      |
| [`pd.core.groupby.Groupby.max`][pdcoregroupbygroupbymax]                               | Compute the maximum value of the group.                                   |
| [`pd.core.groupby.Groupby.mean`][pdcoregroupbygroupbymean]                             | Compute the mean value of the group.                                      |
| [`pd.core.groupby.Groupby.median`][pdcoregroupbygroupbymedian]                         | Compute the median value of the group.                                    |
| [`pd.core.groupby.Groupby.min`][pdcoregroupbygroupbymin]                               | Compute the minimum value of the group.                                   |
| [`pd.core.groupby.DataFrameGroupby.nunique`][pdcoregroupbydataframegroupbynunique]     | Compute the number of unique values in the group.                         |
| [`pd.core.groupby.DataFrameGroupby.ngroup`][pdcoregroupbydataframegroupbyngroup]     | Compute a unique index number for each group.                         |
| [`pd.core.groupby.Groupby.pipe`][pdcoregroupbygroupbypipe]                             | Apply a function to each group.                                           |
| [`pd.core.groupby.Groupby.prod`][pdcoregroupbygroupbyprod]                             | Compute the product of the group.                                         |
| [`pd.core.groupby.Groupby.rolling`][pdcoregroupbygroupbyrolling]                       | Provide rolling window calculations.                                      |
| [`pd.Series.groupby`][pdseriesgroupby]                                                 | Group series using a mapper or by a series of columns.                    |
| [`pd.core.groupby.Groupby.shift`][pdcoregroupbydataframegroupbyshift]                  | Shift the group by a number of periods.                                   |
| [`pd.core.groupby.Groupby.size`][pdcoregroupbygroupbysize]                             | Compute group sizes.                                                      |
| [`pd.core.groupby.Groupby.std`][pdcoregroupbygroupbystd]                               | Compute the standard deviation of the group.                              |
| [`pd.core.groupby.Groupby.sum`][pdcoregroupbygroupbysum]                               | Compute the sum of the group.                                             |
| [`pd.core.groupby.Groupby.transform`][pdcoregroupbydataframegroupbytransform]          | Apply a function to each group.                                           |
| [`pd.core.groupby.SeriesGroupBy.value_counts`][pdcoregroupbyseriesgroupbyvalue_counts] | Count unique values in the group.                                         |
| [`pd.core.groupby.Groupby.var`][pdcoregroupbygroupbyvar]                               | Compute the variance of the group.                                        |



