

PySpark Bodo Cheatsheet   {#pscheatsheet}
========================

References of PySpark methods and their Python equivalents supported by Bodo.





pyspark.sql.SparkSession {#pssession}
-------------------------
The table below is a reference of SparkSession methods and their equivalents in Python,
which are supported by Bodo.


|PySpark Method|Python Equivalent|
|-|-|
|`pyspark.sql.SparkSession.read.csv`|- `pd.read_csv()`|
|`pyspark.sql.SparkSession.read.text`| `pd.read_csv("file.txt", sep="\n", names=["value"], dtype={"value": "str"})`|
|`pyspark.sql.SparkSession.read.parquet`|`pd.read_parquet()`|
|`pyspark.sql.SparkSession.read.json`|`pd.read_json()`|



pyspark.sql.DataFrame   {#psdataframe}
----------------------
The table below is a reference of Spark DataFrame methods and their equivalents in Python,
which are supported by Bodo.




| PySpark Method|Python Equivalent|
|-|-|
| `pyspark.sql.DataFrame.alias`|`alias = df`|
| `pyspark.sql.DataFrame.approxQuantile`|`df[['A', 'B', 'C']].quantile(q)`|
|`pyspark.sql.DataFrame.columns`|`df.columns`|
| `pyspark.sql.DataFrame.corr`|`df[['A', 'B']].corr()`|
| `pyspark.sql.DataFrame.count`|`df.count()`|
| `pyspark.sql.DataFrame.cov`|`df[['A', 'B']].cov()`|
| `pyspark.sql.DataFrame.crossJoin`|`df1.assign(key=1).merge(df2.assign(key=1), on="key").drop("key", axis=1)`|
| `pyspark.sql.DataFrame.describe`|`df.describe()`|
| `pyspark.sql.DataFrame.distinct`|`df.distinct()`|
| `pyspark.sql.DataFrame.drop`|`df.drop(col, axis=1)`|
| `pyspark.sql.DataFrame.dropDuplicates`|`df.drop_duplicates()`|
| `pyspark.sql.DataFrame.drop_duplicates`|`df.drop_duplicates()`|
| `pyspark.sql.DataFrame.dropna`|`df.dropna()`|
| `pyspark.sql.DataFrame.fillna`|`df.fillna(value)`|
| `pyspark.sql.DataFrame.filter`|`df[cond]`|
| `pyspark.sql.DataFrame.first`|`df.head(1)`|
| `pyspark.sql.DataFrame.foreach`|`df.apply(f, axis=1)`|
| `pyspark.sql.DataFrame.groupBy`|`df.groupby("col")`|
| `pyspark.sql.DataFrame.groupby`|`df.groupby("col")`|
| `pyspark.sql.DataFrame.head`|`df.head(n)`|
| `pyspark.sql.DataFrame.intersect`|`pd.merge(df1[['col1', 'col2']].drop_duplicates(), df2[['col1', 'col2']].drop_duplicates(), on =['col1', 'col2'])`|
| `pyspark.sql.DataFrame.intersectAll`|`pd.merge(df1[['col1', 'col2']], df2[['col1', 'col2']].drop_duplicates(), on =['col1', 'col2'])`|
| `pyspark.sql.DataFrame.join`|`df1.join(df2)`|
| `pyspark.sql.DataFrame.orderBy`|`df.sort_values('colname')`|
| `pyspark.sql.DataFrame.show`|`print(df.head(n))`|
| `pyspark.sql.DataFrame.sort`|`df.sort_values('colname')`|


pyspark.sql.functions   {#psfunctions}
----------------------

The table below is a reference of Spark SQL functions and their equivalents in Python,
which are supported by Bodo.


| PySpark Function| Python Equivalent|
|-|-|
| `pyspark.sql.functions.abs`| `df.col.abs()`|
| `pyspark.sql.functions.acos`| `np.arccos(df.col)`|
| `pyspark.sql.functions.acosh`| `np.arccosh(df.col)`|
| `pyspark.sql.functions.add_months`| `df.col + pd.DateOffset(months=num_months)`|
| `pyspark.sql.functions.approx_count_distinct`| `df.col.nunique()`|
| `pyspark.sql.functions.array_contains`| `df.col.apply(lambda a, value: value in a, value=value)`|
| `pyspark.sql.functions.array_distinct`| `df.col.map(lambda x: np.unique(x))`|
| `pyspark.sql.functions.array_except`| `df[['col1', 'col2']].apply(lambda x: np.setdiff1d(x[0], x[1]), axis=1)`|
| `pyspark.sql.functions.array_join`| `df.col.apply(lambda x, sep: sep.join(x), sep=sep)`|
| `pyspark.sql.functions.array_max`| `df.col.map(lambda x: np.nanmax(x))`|
| `pyspark.sql.functions.array_min`| `df.col.map(lambda x: np.nanmin(x))`|
| `pyspark.sql.functions.array_position`| `df.col.apply(lambda x, value: np.append(np.where(x == value)[0], -1)[0], value=value)`|
| `pyspark.sql.functions.array_repeat`| `df.col.apply(lambda x, count: np.repeat(x, count), count=count)`|
| `pyspark.sql.functions.array_sort`| `df.col.map(lambda x: np.sort(x))`|
| `pyspark.sql.functions.array_union`| `df[['col1', 'col2']].apply(lambda x: np.union1d(x[0], x[1]), axis=1)`|
| `pyspark.sql.functions.array_overlap`| `df[['A', 'B']].apply(lambda x: len(np.intersect1d(x[0], x[1])) > 0, axis=1)`|
| `pyspark.sql.functions.asc`| `df.sort_values('col')`|
| `pyspark.sql.functions.asc_nulls_first`| `df.sort_values('col', na_position='first')`|
| `pyspark.sql.functions.asc_nulls_last`| `df.sort_values('col')`|
| `pyspark.sql.functions.ascii`| `df.col.map(lambda x: ord(x[0]))`|
| `pyspark.sql.functions.asin`| `np.arcsin(df.col)`|
| `pyspark.sql.functions.asinh`| `np.arcsinh(df.col)`|
| `pyspark.sql.functions.atan`| `np.arctan(df.col)`|
| `pyspark.sql.functions.atanh`| `np.arctanh(df.col)`|
| `pyspark.sql.functions.atan2`| `df[['col1', 'col2']].apply(lambda x: np.arctan2(x[0], x[1]), axis=1)`|
| `pyspark.sql.functions.avg`| `df.col.mean()`|
| `pyspark.sql.functions.bin`| `df.col.map(lambda x: "{0:b}".format(x))`|
| `pyspark.sql.functions.bitwiseNOT`| `np.invert(df.col)`|
| `pyspark.sql.functions.bround`| `df.col.apply(lambda x, scale: np.round(x, scale), scale=scale)`|
| `pyspark.sql.functions.cbrt`| `df.col.map(lambda x: np.cbrt(x))`|
| `pyspark.sql.functions.ceil`| `np.ceil(df.col)`|
| `pyspark.sql.functions.col`| `df.col`|
| `pyspark.sql.functions.collect_list`| `df.col.to_numpy()`|
| `pyspark.sql.functions.collect_set`| `np.unique(df.col.to_numpy())`|
| `pyspark.sql.functions.column`| `df.col`|
| `pyspark.sql.functions.corr`| `df[['col1', 'col2']].corr(method = 'pearson')`|
| `pyspark.sql.functions.cos`| `np.cos(df.col)`|
| `pyspark.sql.functions.cosh`| `np.cosh(df.col)`|
| `pyspark.sql.functions.count`| `df.col.count()`|
| `pyspark.sql.functions.countDistinct`| `df.col.drop_duplicates().count()`|
| `pyspark.sql.functions.current_date`| `datetime.datetime.now().date()`|
| `pyspark.sql.functions.current_timestamp`| `datetime.datetime.now()`|
| `pyspark.sql.functions.date_add`| `df.col + pd.tseries.offsets.DateOffset(num_days)`|
| `pyspark.sql.functions.date_format`| `df.col.dt.strftime(format_str)`|
| `pyspark.sql.functions.date_sub`| `df.col - pd.tseries.offsets.DateOffset(num_days)`|
| `pyspark.sql.functions.datediff`| `(df.col1 - df.col2).dt.days`|
| `pyspark.sql.functions.dayofmonth`| `df.col.dt.day`|
| `pyspark.sql.functions.dayofweek`| `df.col.dt.dayofweek`|
| `pyspark.sql.functions.dayofyear`| `df.col.dt.dayofyear`|
| `pyspark.sql.functions.degrees`| `np.degrees(df.col)`|
| `pyspark.sql.functions.desc`| `df.sort_values('col', ascending=False)`|
| `pyspark.sql.functions.desc_nulls_first`| `df.sort_values('col', ascending=False, na_position='first')`|
| `pyspark.sql.functions.desc_nulls_last`| `df.sort_values('col', ascending=False)`|
| `pyspark.sql.functions.exp`| `np.exp(df.col)`|
| `pyspark.sql.functions.expm1`| `np.exp(df.col) - 1`|
| `pyspark.sql.functions.factorial`| `df.col.map(lambda x: math.factorial(x))`|
| `pyspark.sql.functions.filter`| `df.filter()`|
| `pyspark.sql.functions.floor`| `np.floor(df.col)`|
| `pyspark.sql.functions.format_number`| `df.col.apply(lambda x,d : ("{:,." + str(d) + "f}").format(np.round(x, d)), d=d)`|
| `pyspark.sql.functions.format_string`| `df.col.apply(lambda x, format_str : format_str.format(x), format_str=format_str)`|
| `pyspark.sql.functions.from_unixtime`| `df.col.map(lambda x: pd.Timestamp(x, 's')).dt.strftime(format_str)`|
| `pyspark.sql.functions.greatest`| `df[['col1', 'col2']].apply(lambda x: np.nanmax(x), axis=1)`|
| `pyspark.sql.functions.hash`| `df.col.map(lambda x: hash(x))`|
| `pyspark.sql.functions.hour`| `df.col.dt.hour`|
| `pyspark.sql.functions.hypot`| `df[['col1', 'col2']].apply(lambda x: np.hypot(x[0], x[1]), axis=1)`|
| `pyspark.sql.functions.initcap`| `df.col.str.title()`|
| `pyspark.sql.functions.instr`| `df.col.str.find(sub=substr)`|
| `pyspark.sql.functions.isnan`| `np.isnan(df.col)`|
| `pyspark.sql.functions.isnull`| `df.col.isna()`|
| `pyspark.sql.functions.kurtosis`| `df.col.kurtosis()`|
| `pyspark.sql.functions.last_day`| `df.col + pd.tseries.offsets.MonthEnd()`|
| `pyspark.sql.functions.least`| `df.min(axis=1)`|
| `pyspark.sql.functions.locate`| `df.col.str.find(sub=substr, start=start)`|
| `pyspark.sql.functions.log`| `np.log(df.col) / np.log(base)`|
| `pyspark.sql.functions.log10`| `np.log10(df.col)`|
| `pyspark.sql.functions.log1p`| `np.log(df.col) + 1`|
| `pyspark.sql.functions.log2`| `np.log2(df.col)`|
| `pyspark.sql.functions.lower`| `df.col.str.lower()`|
| `pyspark.sql.functions.lpad`| `df.col.str.pad(len, flllchar=char)`|
| `pyspark.sql.functions.ltrim`| `df.col.str.lstrip()`|
| `pyspark.sql.functions.max`| `df.col.max()`|
| `pyspark.sql.functions.mean`| `df.col.mean()`|
| `pyspark.sql.functions.min`| `df.col.min()`|
| `pyspark.sql.functions.minute`| `df.col.dt.minute`|
| `pyspark.sql.functions.monotonically_increasing_id`| `pd.Series(np.arange(len(df)))`|
| `pyspark.sql.functions.month`| `df.col.dt.month`|
| `pyspark.sql.functions.nanvl`| `df[['A', 'B']].apply(lambda x: x[0] if not pd.isna(x[0]) else x[1], axis=1)`|
| `pyspark.sql.functions.overlay`| `df.A.str.slice_replace(start=index, stop=index+len, repl=repl_str)`|
| `pyspark.sql.functions.pandas_udf`| `df.apply(f)` or `df.col.map(f)`|
| `pyspark.sql.functions.pow`| `np.power(df.col1, df.col2)`|
| `pyspark.sql.functions.quarter`| `df.col.dt.quarter`|
| `pyspark.sql.functions.radians`| `np.radians(df.col)`|
| `pyspark.sql.functions.rand`| `pd.Series(np.random.rand(1, num_cols))`|
| `pyspark.sql.functions.randn`| `pd.Series(np.random.randn(num_cols))`|
| `pyspark.sql.functions.regexp_replace`| `df.col.str.replace(pattern, repl_string)`|
| `pyspark.sql.functions.repeat`| `df.col.str.repeat(count)`|
| `pyspark.sql.functions.reverse`| `df.col.map(lambda x: x[::-1])`|
| `pyspark.sql.functions.rint`| `df.col.map(lambda x: int(np.round(x, 0)))`|
| `pyspark.sql.functions.round`| `df.col.apply(lambda x, decimal_places: np.round(x, decimal_places), decimal_places=decimal_places)`|
| `pyspark.sql.functions.rpad`| `df.col.str.pad(len, side='right', flllchar=char)`|
| `pyspark.sql.functions.rtrim`| `df.col.str.rstrip()`|
| `pyspark.sql.functions.second`| `df.col.dt.second`|
| `pyspark.sql.functions.sequence`| `df[['col1', 'col2', 'col3']].apply(lambda x: np.arange(x[0], x[1], x[2]), axis=1)`|
| `pyspark.sql.functions.shuffle`| `df.col.map(lambda x: np.random.permutation(x))`|
| `pyspark.sql.functions.signum`| `np.sign(df.col)`|
| `pyspark.sql.functions.sin`| `np.sin(df.col)`|
| `pyspark.sql.functions.sinh`| `np.sinh(df.col)`|
| `pyspark.sql.functions.size`| `df.col.map(lambda x: len(x))`|
| `pyspark.sql.functions.skewness`| `df.col.skew()`|
| `pyspark.sql.functions.slice`| `df.col.map(lambda x: x[start : end])`|
| `pyspark.sql.functions.split`| `df.col.str.split(pat, num_splits)`|
| `pyspark.sql.functions.sqrt`| `np.sqrt(df.col)`|
| `pyspark.sql.functions.stddev`| `df.col.std()`|
| `pyspark.sql.functions.stddev_pop`| `df.col.std(ddof=0)`|
| `pyspark.sql.functions.stddev_samp`| `df.col.std()`|
| `pyspark.sql.functions.substring`| `df.col.str.slice(start, start+len)`|
| `pyspark.sql.functions.substring_index`| `df.col.apply(lambda x, sep, count: sep.join(x.split(sep)[:count]), sep=sep, count=count)`|
| `pyspark.sql.functions.sum`| `df.col.sum()`|
| `pyspark.sql.functions.sumDistinct`| `df.col.drop_duplicates().sum()`|
| `pyspark.sql.functions.tan`| `np.tan(df.col)`|
| `pyspark.sql.functions.tanh`| `np.tanh(df.col)`|
| `pyspark.sql.functions.timestamp_seconds`| `pd.to_datetime("now")`|
| `pyspark.sql.functions.to_date`| `df.col.apply(lambda x, format_str: pd.to_datetime(x, format=format_str).date(), format_str=format_str)`|
| `pyspark.sql.functions.to_timestamp`| `df.A.apply(lambda x, format_str: pd.to_datetime(x, format=format_str), format_str=format_str)`|
| `pyspark.sql.functions.translate`| `df.col.str.split("").apply(lambda x: "".join(pd.Series(x).replace(to_replace, values).tolist()), to_replace=to_replace, values=values)`|
| `pyspark.sql.functions.trim`| `df.col.str.strip()`|
| `pyspark.sql.functions.udf`| `df.apply` or `df.col.map`|
| `pyspark.sql.functions.unix_timestamp`| `df.col.apply(lambda x, format_str: (pd.to_datetime(x, format=format_str) - pd.Timestamp("1970-01-01")).total_seconds(), format_str=format_str)`|
| `pyspark.sql.functions.upper`| `df.col.str.upper()`|
| `pyspark.sql.functions.var_pop`| `df.col.var(ddof=0)`|
| `pyspark.sql.functions.var_samp`| `df.col.var()`|
| `pyspark.sql.functions.variance`| `df.col.var()`|
| `pyspark.sql.functions.weekofyear`| `df.col.dt.isocalendar().week`|
| `pyspark.sql.functions.when`| `df.A.apply(lambda a, cond, val, other: val if cond(a) else other, cond=cond, val=val, other=other)`|
| `pyspark.sql.functions.year`| `df.col.dt.year`|

### Special Cases

#### `pyspark.sql.functions.concat`
-   `pyspark.sql.functions.concat`
    - for Arrays : `df[['col1', 'col2', 'col3']].apply(lambda x: np.hstack(x), axis=1)`
    - for Strings : `df[['col1', 'col2', 'col3']].apply(lambda x: "".join(x), axis=1)`

#### `pyspark.sql.functions.conv`
-   `pyspark.sql.functions.conv`
    
    pandas equivalent: 
    
    ```py
    base_map = {2: "{0:b}", 8: "{0:o}", 10: "{0:d}", 16: "{0:x}"}
    new_format = base_map[new_base]
    df.col.apply(lambda x, old_base, new_format: new_format.format(int(x, old_base)), old_base=old_base, new_format=new_format)
    ```
    
####  `pyspark.sql.functions.date_trunc`

-   `pyspark.sql.functions.date_trunc`   

    - For frequencies day and below: `df.col.dt.floor(freq=trunc_val)`
    - For month: `df.col.map(lambda x: pd.Timestamp(year=x.year, month=x.month, day=1))`
    - For year: `df.col.map(lambda x: pd.Timestamp(year=x.year, month=1, day=1))`

####   `pyspark.sql.functions.regexp_extract`

-   `pyspark.sql.functions.regexp_extract`

    Here's a small pandas function equivalent:
    
    ```py
    def f(x, pat):
        res = re.search(pat, x)
        return "" if res is None else res[0]
    df.col.apply(f, pat=pat)  
    ```         
    
#### `pyspark.sql.functions.shiftLeft`
-   `pyspark.sql.functions.shiftLeft`

    - If the type is uint64 `np.left_shift(df.col.astype(np.int64), numbits).astype(np.uint64))`
    - Other integer types: `np.left_shift(df.col, numbits)`

####  `pyspark.sql.functions.shiftRight`

- `pyspark.sql.functions.shiftRight`

    - If the type is uint64 use `shiftRightUnsigned`
    - Other integer types: `np.right_shift(df.col, numbits)`
    
#### `pyspark.sql.functions.shiftRightUnsigned`

- `pyspark.sql.functions.shiftRightUnsigned`

    Here's a small pandas function equivalent:

    ```py 
    def shiftRightUnsigned(col, num_bits):
        bits_minus_1 = max((num_bits - 1), 0)
        mask_bits = (np.int64(1) << bits_minus_1) - 1
        mask = ~(mask_bits << (63 - bits_minus_1))
        return np.right_shift(col.astype(np.int64), num_bits) & mask).astype(np.uint64)
    shiftRightUnsigned(df.col, numbits)  
    ```    
  
#### `pyspark.sql.functions.sort_array`
-   `pyspark.sql.functions.sort_array`

    - Ascending:  `df.col.map(lambda x: np.sort(x))`
    - Descending: `df.col.map(lambda x: np.sort(x)[::-1])`
    
    
#### `pyspark.sql.functions.trunc`
-   `pyspark.sql.functions.trunc`

    ```py
    def f(date, trunc_str):
        if trunc_str == 'year':
            return pd.Timestamp(year=date.year, month=1, day=1)
        if trunc_str == 'month':
            return pd.Timestamp(year=date.year, month=date.month, day=1)
    df.A.apply(f, trunc_str=trunc_str)
    ```