.. _spark:

Migration from Spark
--------------------

This section provides information for migrating Spark workloads to Bodo, unlocking
orders of magnitude performance improvement, as well as simplicity and maintainability of Python codes.
Spark APIs are usually equivalent to simpler Python/Pandas APIs, which are automatically parallelized by Bodo.


pyspark.sql.DataFrame
~~~~~~~~~~~~~~~~~~~~~
The table below is a reference of Spark DataFrame methods and their equivalents in Python, 
which are supported by Bodo.

.. list-table::
  :header-rows: 1

  * - Pyspark Method
    - Python Equivalent
  * - :meth:`pyspark.sql.DataFrame.alias`
    - ``alias = df``
  * - :meth:`pyspark.sql.DataFrame.approxQuantile`
    - ``df[['A', 'B', 'C']].quantile(q)``
  * - :attr:`pyspark.sql.DataFrame.columns`
    - ``df.columns``
  * - :meth:`pyspark.sql.DataFrame.corr`
    - ``df[['A', 'B']].corr()``
  * - :meth:`pyspark.sql.DataFrame.count`
    - ``df.count()``
  * - :meth:`pyspark.sql.DataFrame.cov`
    - ``df[['A', 'B']].cov()``
  * - :meth:`pyspark.sql.DataFrame.crossJoin`
    - ``df1.assign(key=1).merge(df2.assign(key=1), on="key").drop("key", axis=1)``
  * - :meth:`pyspark.sql.DataFrame.describe`
    - ``df.describe()``
  * - :meth:`pyspark.sql.DataFrame.distinct`
    - ``df.distinct()``
  * - :meth:`pyspark.sql.DataFrame.drop`
    - ``df.drop(col, axis=1)``
  * - :meth:`pyspark.sql.DataFrame.dropDuplicates`
    - ``df.drop_duplicates()``
  * - :meth:`pyspark.sql.DataFrame.drop_duplicates`
    - ``df.drop_duplicates()``
  * - :meth:`pyspark.sql.DataFrame.dropna`
    - ``df.dropna()``
  * - :meth:`pyspark.sql.DataFrame.fillna`
    - ``df.fillna(value)``
  * - :meth:`pyspark.sql.DataFrame.filter`
    - ``df[cond]``
  * - :meth:`pyspark.sql.DataFrame.first`
    - ``df.head(1)``
  * - :meth:`pyspark.sql.DataFrame.foreach`
    - ``df.apply(f, axis=1)``
  * - :meth:`pyspark.sql.DataFrame.groupBy`
    - ``df.groupby("col")``
  * - :meth:`pyspark.sql.DataFrame.groupby`
    - ``df.groupby("col")``
  * - :meth:`pyspark.sql.DataFrame.head`
    - ``df.head(n)``
  * - :meth:`pyspark.sql.DataFrame.intersect`
    - ``pd.merge(df1[['col1', 'col2']].drop_duplicates(), df2[['col1', 'col2']].drop_duplicates(), on =['col1', 'col2'])``
  * - :meth:`pyspark.sql.DataFrame.intersectAll`
    - ``pd.merge(df1[['col1', 'col2']], df2[['col1', 'col2']].drop_duplicates(), on =['col1', 'col2'])``
  * - :meth:`pyspark.sql.DataFrame.join`
    - ``df1.join(df2)``
  * - :meth:`pyspark.sql.DataFrame.orderBy`
    - ``df.sort_values('colname')``
  * - :meth:`pyspark.sql.DataFrame.show`
    - ``print(df.head(n))``
  * - :meth:`pyspark.sql.DataFrame.sort`
    - ``df.sort_values('colname')``


pyspark.sql.functions
~~~~~~~~~~~~~~~~~~~~~

The table below is a reference of Spark SQL functions and their equivalents in Python, 
which are supported by Bodo.

.. list-table::
  :header-rows: 1

  * - Pyspark Function
    - Python Equivalent
  * - :func:`pyspark.sql.functions.abs`
    - ``df.col.abs()``
  * - :func:`pyspark.sql.functions.acos`
    - ``np.arccos(df.col)``
  * - :func:`pyspark.sql.functions.add_months`
    - ``df.col + pd.DateOffset(months=num_months)``
  * - :func:`pyspark.sql.functions.approx_count_distinct`
    - ``df.col.nunique()``
  * - :func:`pyspark.sql.functions.array_distinct`
    - ``df.col.map(lambda x: np.unique(x))``
  * - :func:`pyspark.sql.functions.array_max`
    - ``df.col.map(lambda x: np.nanmax(x))``
  * - :func:`pyspark.sql.functions.array_min`
    - ``df.col.map(lambda x: np.nanmin(x))``
  * - :func:`pyspark.sql.functions.array_position`
    - | ``df.col.apply(lambda x, value: np.append(np.where(x == value)[0], -1)[0], value=value)``
      | (Note, Python uses 0 indexing)
  * - :func:`pyspark.sql.functions.array_repeat`
    - ``df.col.apply(lambda x, count: np.repeat(x, count), count=count)``
  * - :func:`pyspark.sql.functions.array_union`
    - ``df[['col1', 'col2']].apply(lambda x: np.union1d(x[0], x[1]), axis=1)``
  * - :func:`pyspark.sql.functions.asc`
    - ``df.sort_values('col')``
  * - :func:`pyspark.sql.functions.asc_nulls_first`
    - ``df.sort_values('col', na_position='first')``
  * - :func:`pyspark.sql.functions.asc_nulls_last`
    - ``df.sort_values('col')``
  * - :func:`pyspark.sql.functions.ascii`
    - ``df.col.map(lambda x: ord(x[0]))``
  * - :func:`pyspark.sql.functions.asin`
    - ``np.arcsin(df.col)``
  * - :func:`pyspark.sql.functions.atan`
    - ``np.arctan(df.col)``
  * - :func:`pyspark.sql.functions.atan2`
    - ``df[['col1', 'col2']].apply(lambda x: np.arctan2(x[0], x[1]), axis=1)``
  * - :func:`pyspark.sql.functions.avg`
    - ``df.col.mean()``
  * - :func:`pyspark.sql.functions.bin`
    - ``df.col.map(lambda x: "{0:b}".format(x))``
  * - :func:`pyspark.sql.functions.bitwiseNOT`
    - ``np.invert(df.col)``
  * - :func:`pyspark.sql.functions.bround`
    - ``df.col.apply(lambda x, scale: np.round(x, scale), scale=scale)``
  * - :func:`pyspark.sql.functions.cbrt`
    - ``df.col.map(lambda x: np.cbrt(x))``
  * - :func:`pyspark.sql.functions.ceil`
    - ``np.ceil(df.col)``
  * - :func:`pyspark.sql.functions.col`
    - ``df.col``
  * - :func:`pyspark.sql.functions.collect_list`
    - ``df.col.to_numpy()``
  * - :func:`pyspark.sql.functions.collect_set`
    - ``np.unique(df.col.to_numpy())``
  * - :func:`pyspark.sql.functions.column`
    - ``df.col``
  * - :func:`pyspark.sql.functions.concat`
    - | # Arrays ``df[['col1', 'col2', 'col3']].apply(lambda x: np.hstack(x), axis=1)``
      | # Strings ``df[['col1', 'col2', 'col3']].apply(lambda x: "".join(x), axis=1)``
  * - :func:`pyspark.sql.functions.concat_ws`
    - ``df[['col1', 'col2', 'col3']].apply(lambda x, sep: sep.join(x), axis=1, sep=sep)``
  * - :func:`pyspark.sql.functions.conv`
    - | ``base_map = {2: "{0:b}", 8: "{0:o}", 10: "{0:d}", 16: "{0:x}"}``
      | ``new_format = base_map[new_base]``
      | ``df.col.apply(lambda x, old_base, new_format: new_format.format(int(x, old_base)), old_base=old_base, new_format=new_format)``
  * - :func:`pyspark.sql.functions.cos`
    - ``np.cos(df.col)``
  * - :func:`pyspark.sql.functions.cosh`
    - ``np.cosh(df.col)``
  * - :func:`pyspark.sql.functions.count`
    - ``df.col.count()``
  * - :func:`pyspark.sql.functions.countDistinct`
    - ``df.col.drop_duplicates().count()``
  * - :func:`pyspark.sql.functions.current_date`
    - ``datetime.datetime.now().date()``
  * - :func:`pyspark.sql.functions.current_timestamp`
    - ``datetime.datetime.now()``
  * - :func:`pyspark.sql.functions.date_add`
    - ``df.col + pd.tseries.offsets.DateOffset(num_days)``
  * - :func:`pyspark.sql.functions.date_format`
    - ``df.col.dt.strftime(format_str)``
  * - :func:`pyspark.sql.functions.date_sub`
    - ``df.col - pd.tseries.offsets.DateOffset(num_days)``
  * - :func:`pyspark.sql.functions.date_trunc`
    - | For frequencies day and below ``df.col.dt.floor(freq=trunc_val)``
      | For month: ``df.col.map(lambda x: pd.Timestamp(year=x.year, month=x.month, day=1))``
      | For year: ``df.col.map(lambda x: pd.Timestamp(year=x.year, month=1, day=1))``
  * - :func:`pyspark.sql.functions.datediff`
    - ``(df.col1 - df.col2).dt.days``
  * - :func:`pyspark.sql.functions.dayofmonth`
    - ``df.col.dt.day``
  * - :func:`pyspark.sql.functions.dayofweek`
    - ``df.col.dt.dayofweek``
  * - :func:`pyspark.sql.functions.dayofyear`
    - ``df.col.dt.dayofyear``
  * - :func:`pyspark.sql.functions.degrees`
    - ``np.degrees(df.col)``
  * - :func:`pyspark.sql.functions.desc`
    - ``df.sort_values('col', ascending=False)``
  * - :func:`pyspark.sql.functions.desc_nulls_first`
    - ``df.sort_values('col', ascending=False, na_position='first')``
  * - :func:`pyspark.sql.functions.desc_nulls_last`
    - ``df.sort_values('col', ascending=False)``
  * - :func:`pyspark.sql.functions.exp`
    - ``np.exp(df.col)``
  * - :func:`pyspark.sql.functions.expm1`
    - ``np.exp(df.col) - 1``
  * - :func:`pyspark.sql.functions.factorial`
    - ``df.col.map(lambda x: math.factorial(x))``
  * - :func:`pyspark.sql.functions.floor`
    - ``np.floor(df.col)``
  * - :func:`pyspark.sql.functions.format_number`
    - ``df.col.apply(lambda x,d : ("{:,." + str(d) + "f}").format(np.round(x, d)), d=d)``
  * - :func:`pyspark.sql.functions.format_string`
    - ``df.col.apply(lambda x, format_str : format_str.format(x), format_str=format_str)``
  * - :func:`pyspark.sql.functions.from_unixtime`
    - ``df.col.map(lambda x: pd.Timestamp(x, 's')).dt.strftime(format_str)``
  * - :func:`pyspark.sql.functions.hash`
    - ``df.col.map(lambda x: hash(x))``
  * - :func:`pyspark.sql.functions.hour`
    - ``df.col.dt.hour``
  * - :func:`pyspark.sql.functions.hypot`
    - ``df[['col1', 'col2']].apply(lambda x: np.hypot(x[0], x[1]), axis=1)``
  * - :func:`pyspark.sql.functions.initcap`
    - ``df.col.str.title()``
  * - :func:`pyspark.sql.functions.instr`
    - ``df.col.str.find(sub=substr)``
  * - :func:`pyspark.sql.functions.isnan`
    - ``np.isnan(df.col)``
  * - :func:`pyspark.sql.functions.isnull`
    - ``df.col.isna()``
  * - :func:`pyspark.sql.functions.kurtosis`
    - ``df.col.kurtosis()``
  * - :func:`pyspark.sql.functions.last_day`
    - ``df.col + pd.tseries.offsets.MonthEnd()``
  * - :func:`pyspark.sql.functions.least`
    - ``df.min(axis=1)``
  * - :func:`pyspark.sql.functions.locate`
    - ``df.col.str.find(sub=substr, start=start)``
  * - :func:`pyspark.sql.functions.log`
    - ``np.log(df.col) / np.log(base)``
  * - :func:`pyspark.sql.functions.log10`
    - ``np.log10(df.col)``
  * - :func:`pyspark.sql.functions.log1p`
    - ``np.log(df.col) + 1``
  * - :func:`pyspark.sql.functions.log2`
    - ``np.log2(df.col)``
  * - :func:`pyspark.sql.functions.lower`
    - ``df.col.str.lower()``
  * - :func:`pyspark.sql.functions.lpad`
    - ``df.col.str.pad(len, flllchar=char)``
  * - :func:`pyspark.sql.functions.ltrim`
    - ``df.col.str.lstrip()``
  * - :func:`pyspark.sql.functions.max`
    - ``df.col.max()``
  * - :func:`pyspark.sql.functions.mean`
    - ``df.col.mean()``
  * - :func:`pyspark.sql.functions.min`
    - ``df.col.min()``
  * - :func:`pyspark.sql.functions.minute`
    - ``df.col.dt.minute``
  * - :func:`pyspark.sql.functions.monotonically_increasing_id`
    - ``pd.Series(np.arange(len(df)))``
  * - :func:`pyspark.sql.functions.month`
    - ``df.col.dt.month``
  * - :func:`pyspark.sql.functions.nanvl`
    - ``df[['A', 'B']].apply(lambda x: x[0] if not pd.isna(x[0]) else x[1], axis=1)``
  * - :func:`pyspark.sql.functions.overlay`
    - ``df.A.str.slice_replace(start=index, stop=index+len, repl=repl_str)``
  * - :func:`pyspark.sql.functions.pandas_udf`
    - ``df.apply(f)`` or ``df.col.map(f)``
  * - :func:`pyspark.sql.functions.pow`
    - ``np.power(df.col1, df.col2)``
  * - :func:`pyspark.sql.functions.quarter`
    - ``df.col.dt.quarter``
  * - :func:`pyspark.sql.functions.radians`
    - ``np.radians(df.col)``
  * - :func:`pyspark.sql.functions.rand`
    - ``pd.Series(np.random.rand(1, num_cols))``
  * - :func:`pyspark.sql.functions.randn`
    - ``pd.Series(np.random.randn(num_cols))``
  * - :func:`pyspark.sql.functions.regexp_extract`
    - | ``def f(x, pat):``
      |     ``res = re.search(pat, x)``
      |     ``return "" if res is None else res[0]``
      | ``df.col.apply(f, pat=pat)``
  * - :func:`pyspark.sql.functions.regexp_replace`
    - ``df.col.str.replace(pattern, repl_string)``
  * - :func:`pyspark.sql.functions.repeat`
    - ``df.col.str.repeat(count)``
  * - :func:`pyspark.sql.functions.reverse`
    - ``df.col.map(lambda x: x[::-1])``
  * - :func:`pyspark.sql.functions.rint`
    - ``df.col.map(lambda x: int(np.round(x, 0)))``
  * - :func:`pyspark.sql.functions.round`
    - ``df.col.apply(lambda x, decimal_places: np.round(x, decimal_places), decimal_places=decimal_places)``
  * - :func:`pyspark.sql.functions.rpad`
    - ``df.col.str.pad(len, side='right', flllchar=char)``
  * - :func:`pyspark.sql.functions.rtrim`
    - ``df.col.str.rstrip()``
  * - :func:`pyspark.sql.functions.second`
    - ``df.col.dt.second``
  * - :func:`pyspark.sql.functions.sequence`
    - ``df[['col1', 'col2', 'col3']].apply(lambda x: np.arange(x[0], x[1], x[2]), axis=1)`` 
  * - :func:`pyspark.sql.functions.shiftLeft`
    - ``np.left_shift(df.col, numbits)``
  * - :func:`pyspark.sql.functions.shuffle`
    - ``df.col.map(lambda x: np.random.permutation(x))`` 
  * - :func:`pyspark.sql.functions.signum`
    - ``np.sign(df.col)`` 
  * - :func:`pyspark.sql.functions.sin`
    - ``np.sin(df.col)``
  * - :func:`pyspark.sql.functions.sinh`
    - ``np.sinh(df.col)``
  * - :func:`pyspark.sql.functions.size`
    - ``df.col.map(lambda x: len(x))``
  * - :func:`pyspark.sql.functions.skewness`
    - ``df.col.skew()``
  * - :func:`pyspark.sql.functions.slice`
    - ``df.col.map(lambda x: x[start : end])``
  * - :func:`pyspark.sql.functions.sort_array`
    - | Ascending:  ``df.col.map(lambda x: np.sort(x))`` 
      | Descending: ``df.col.map(lambda x: np.sort(x)[::-1])``
  * - :func:`pyspark.sql.functions.split`
    - ``df.col.str.split(pat, num_splits)``
  * - :func:`pyspark.sql.functions.sqrt`
    - ``np.sqrt(df.col)`` 
  * - :func:`pyspark.sql.functions.stddev`
    - ``df.col.std()``
  * - :func:`pyspark.sql.functions.stddev_pop`
    - ``df.col.std(ddof=0)`` 
  * - :func:`pyspark.sql.functions.stddev_samp`
    - ``df.col.std()`` 
  * - :func:`pyspark.sql.functions.substring`
    - ``df.col.str.slice(start, start+len)``
  * - :func:`pyspark.sql.functions.substring_index`
    - ``df.col.apply(lambda x, sep, count: sep.join(x.split(sep)[:count]), sep=sep, count=count)``
  * - :func:`pyspark.sql.functions.sum`
    - ``df.col.sum()``
  * - :func:`pyspark.sql.functions.sumDistinct`
    - ``df.col.drop_duplicates().sum()``
  * - :func:`pyspark.sql.functions.tan`
    - ``np.tan(df.col)``
  * - :func:`pyspark.sql.functions.tanh`
    - ``np.tanh(df.col)`` 
  * - :func:`pyspark.sql.functions.translate`
    - ``df.col.str.split("").apply(lambda x: "".join(pd.Series(x).replace(to_replace, values).tolist()), to_replace=to_replace, values=values)``
  * - :func:`pyspark.sql.functions.trim`
    - ``df.col.str.strip()``
  * - :func:`pyspark.sql.functions.udf`
    - ``df.apply`` or ``df.col.map`` 
  * - :func:`pyspark.sql.functions.upper`
    - ``df.col.str.upper()``
  * - :func:`pyspark.sql.functions.var_pop`
    - ``df.col.var(ddof=0)`` 
  * - :func:`pyspark.sql.functions.var_samp`
    - ``df.col.var()`` 
  * - :func:`pyspark.sql.functions.variance`
    - ``df.col.var()``
  * - :func:`pyspark.sql.functions.weekofyear`
    - ``df.col.dt.isocalendar().week``
  * - :func:`pyspark.sql.functions.when`
    - ``df.A.apply(lambda a, cond, val, other: val if cond(a) else other, cond=cond, val=val, other=other)``
  * - :func:`pyspark.sql.functions.year`
    - ``df.col.dt.year``
