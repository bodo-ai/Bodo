.. _spark:

Migration from Spark
============


Examples
--------------------

Bodo offers simplicity and maintainability of Python codes while unlocking orders of magnitude performance improvement. Spark APIs are usually equivalent to simpler Python/Pandas APIs, which are automatically parallelized by Bodo. This section aims to assist spark users with their transition to Bodo. Here, we show the most common data wrangling methods in Pyspark and Pandas through brief code examples. We used the COVID-19 World Vaccination Progress dataset that can be downloaded from `Kaggle <https://www.kaggle.com/gpreda/covid-world-vaccination-progress?select=country_vaccinations.csv>`_. If you want to execute the code as shown below, make sure that you have `Bodo <https://docs.bodo.ai/latest/source/install.html>`_ installed. Here is a list of examples:

#. :ref:`Environment Setup`
#. :ref:`Load Data`
#. :ref:`Display the Schema of the DataFrame`
#. :ref:`Change Data Types of the DataFrame`
#. :ref:`Display the Head of the DataFrame`
#. :ref:`Select Columns from the DataFrame` 
#. :ref:`Show the Statistics of the DataFrame` 
#. :ref:`Drop Duplicate Values`
#. :ref:`Missing Values` 
#. :ref:`DateTime Manipulation`
#. :ref:`Filter Data Based on Conditions`
#. :ref:`Aggregation Functions`
#. :ref:`Sort Data`
#. :ref:`Rename Columns`
#. :ref:`Create New Columns`
#. :ref:`User-Defined Functions`
#. :ref:`Create a DataFrame`
#. :ref:`Export the Data`


.. _Environment Setup:

Environment Setup 
~~~~~~~~~~~~~~~~~~~~~
With Bodo::

    import bodo
    import pandas as pd
    import numpy as np 


With PySpark::

    from pyspark.sql import SparkSession
    spark = SparkSession \
        .builder \
        .appName("Migration From Spark") \
        .getOrCreate()

.. _Load Data:

Load Data 
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def load_data():
	df = pd.read_csv('country_vaccinations_by_manufacturer.csv')
    	return df

    df = load_data()


With PySpark::	

    data = spark.read.csv('country_vaccinations_by_manufacturer.csv',header = True)


.. _Display the Schema of the DataFrame:

Display the Schema of the DataFrame
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def schema(df):
        print(df.dtypes)

    schema(df)

With PySpark::

    print(data.printSchema())


.. _Change Data Types of the DataFrame:

Change Data Types of the DataFrame
~~~~~~~~~~~~~~~~~~~~~
 
With Bodo::

    @bodo.jit(distributed = ['df'])
    def load_data():
        df = pd.read_csv('country_vaccinations_by_manufacturer.csv', 
                         dtype = {'location' : 'str', 'vaccine' : 'str',
                                  'total_vaccinations' : 'Int64'}, 
                         parse_dates=['date'])
        print(df.info())
        return df

    df = load_data()

With PySpark::

    from pyspark.sql.types import StructField,IntegerType, StringType, DateType, StructType

    new_schema = [StructField('location', StringType(), True),
                  StructField('date', DateType(), True), 
                  StructField('vaccine', StringType(), True),
                  StructField('total_vaccinations', IntegerType(), True)]

    data = spark.read.csv('country_vaccinations_by_manufacturer.csv',header = True, 
                      schema = StructType(fields = new_schema))
    data.printSchema()


.. _Display the Head of the DataFrame:

Display the Head of the DataFrame
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def head_data(df):
    	print(df.head())

    head_data(df)

With PySpark::

    data.show(5)
    data.take(5)


.. _Select Columns from the DataFrame:

Select Columns from the DataFrame
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df', 'df_columns'])
    def load_data(df):
	df_columns = df[['location', 'vaccine']]
    	return df_columns

    df_columns = load_data(df)

With PySpark::

    data_columns = data.select('location', 'vaccine').show()


.. _Show the Statistics of the DataFrame:

Show the Statistics of the DataFrame
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def get_describe(df):
    	print(df.describe())

    get_describe(df)

With PySpark::

    data.describe().show()


.. _Drop Duplicate Values:

Drop Duplicate Values
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df', 'df_cleaned'])
    def drop(df):
    	df_cleaned = df.drop_duplicates()
    	return df_cleaned

    df_cleaned = drop(df)

With PySpark::

    data.dropDuplicates().show()


.. _Missing Values:

Missing Values (Count NA, Drop NA, Replace NA)
~~~~~~~~~~~~~~~~~~~~~

| Count NA :

With Bodo::

    @bodo.jit(distributed = ['df'])
    def count_na(df):
    	print(df.isnull().sum())

    count_na(df)

With PySpark::

    from pyspark.sql.functions import isnan, when, count, col

    data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_s.columns]).show()

Drop NA :

With Bodo::

    @bodo.jit(distributed = ['df', 'df_valid'])
    def drop_na(df):
    	df_valid = df.dropna(how ='any')
    	return df_valid

    df_valid = drop_na(df)

With PySpark::

    data_valid = data.dropna(how='any')

Replace NA :

With Bodo::

    @bodo.jit(distributed = ['df', 'df_filled'])
    def replace_na(df):
    	df_filled = df.fillna(0)
    	return df_filled

    df_filled = replace_na(df)

With PySpark::

    data_replaced = data.na.fill(value = 0)


.. _DateTime Manipulation:

DateTime Manipulation
~~~~~~~~~~~~~~~~~~~~~

| Convert String to Datetime : 

With Bodo::

    @bodo.jit(distributed = ['df'])
    def convert_date(df):
    	df['record_date'] = pd.to_datetime(df['date'])
    	return df

    df = convert_date(df)

With PySpark::

    from pyspark.sql.types import DateType

    data = data.withColumn("record_date", data["date"].cast(DateType()))


Extract Day / Month / Year from Datetime : 

With Bodo::

    @bodo.jit(distributed = ['df'])
    def extract_date(df):
    	print(df['record_date'].dt.year)

    extract_date(df)

With PySpark::

    from pyspark.sql.functions import year

    data.select(year(df_s.record_date)).show()


.. _Filter Data Based on Conditions:

Filter Data Based on Conditions
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df', 'df_filtered'])
    def sort_data(df):
    	df_filtered = df[df.vaccine =='Pfizer/BioNTech']
    	return df_filtered

    df_filtered = sort_data(df)

With PySpark::

    data_filtered = data.where(data.vaccine =='Pfizer/BioNTech')


.. _Aggregation Functions:

Aggregation Functions: (sum, count, mean, max, min, etc)
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def group_by(df):
    	print(df.groupby('location').agg({'total_vaccinations' : 'sum'}))

    group_by(df)

With PySpark::

    data.groupBy('location').agg({'total_vaccinations' : 'sum'}).show()


.. _Sort Data:

Sort Data
~~~~~~~~~~~~~~~~~~~~~ 

With Bodo::

    @bodo.jit(distributed = ['df', 'df_sorted'])
    def sort_data(df):
        df_sorted = df.sort_values(by = ['total_vaccinations'], ascending=False)
        return df_sorted

    df_sorted = sort_data(df)

With PySpark::

    from pyspark.sql.types import IntegerType
    from pyspark.sql.functions import col
    from pyspark.sql.functions import desc 

    data_sorted = data.withColumn("total_vaccinations", col("total_vaccinations") 
	              .cast(IntegerType())).select("total_vaccinations") 
                      .sort(desc('total_vaccinations')).show()


.. _Rename Columns:

Rename Columns
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df', 'df_renamed'])
    def rename_column(df):
    	df_renamed = df.rename(columns = {'location' : 'country'}, inplace = True)
    
    	return data_renamed

    df_renamed = rename_column(df)

With PySpark::

    data_renamed = data.withColumnRenamed("location","country").show()


.. _Create New Columns:

Create New Columns
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def create_column(df):
    	df['doubled'] = 2 * df['total_vaccinations']
    	return df

    df = create_column(df)

With PySpark::

    from pyspark.sql.functions import col

    data = data.withColumn("doubled", 2*col("total_vaccinations")).show()


.. _User-Defined Functions:

User-Defined Functions
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def udf(df):
        df['new_column'] = df['location'].apply(lambda x: x.upper())
        return df

    df = udf(df)

With PySpark::

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    pyspark_udf = udf(lambda x: x.upper(), StringType())
    data = data.withColumn("new_column", pyspark_udf(df_s.location)).show()


.. _Create a DataFrame:

Create a DataFrame 
~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit()
    def create():
    	df = pd.DataFrame({'id': [1, 2], 'label': ["one", "two"]})
    	return df

    df = create()

With PySpark::

    data = spark.createDataFrame([(1, "one"),(2, "two"),],["id", "label"])


.. _Export the Data:

Export the Data
~~~~~~~~~~~~~~~~~~~~~ 

With Bodo::

    @bodo.jit()
    def export_data():
        df = pd.DataFrame({'id': [1, 2], 'label': ["one", "two"]})
        df_pandas = df.to_csv('pandas_data.csv')
        return df_pandas

    export_data()

With PySpark::

    df = spark.createDataFrame([(1, "one"),(2, "two"),],["id", "label"])
    df_spark.write.csv("df_spark.csv", header = True)


pyspark.sql.DataFrame
--------------------
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
--------------------

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
  * - :func:`pyspark.sql.functions.acosh`
    - ``np.arccosh(df.col)``
  * - :func:`pyspark.sql.functions.add_months`
    - ``df.col + pd.DateOffset(months=num_months)``
  * - :func:`pyspark.sql.functions.approx_count_distinct`
    - ``df.col.nunique()``
  * - :func:`pyspark.sql.functions.array_contains`
    - ``df.col.apply(lambda a, value: value in a, value=value)``
  * - :func:`pyspark.sql.functions.array_distinct`
    - ``df.col.map(lambda x: np.unique(x))``
  * - :func:`pyspark.sql.functions.array_except`
    - ``df[['col1', 'col2']].apply(lambda x: np.setdiff1d(x[0], x[1]), axis=1)``
  * - :func:`pyspark.sql.functions.array_join`
    - ``df.col.apply(lambda x, sep: sep.join(x), sep=sep)``
  * - :func:`pyspark.sql.functions.array_max`
    - ``df.col.map(lambda x: np.nanmax(x))``
  * - :func:`pyspark.sql.functions.array_min`
    - ``df.col.map(lambda x: np.nanmin(x))``
  * - :func:`pyspark.sql.functions.array_position`
    - | ``df.col.apply(lambda x, value: np.append(np.where(x == value)[0], -1)[0], value=value)``
      | (Note, Python uses 0 indexing)
  * - :func:`pyspark.sql.functions.array_repeat`
    - ``df.col.apply(lambda x, count: np.repeat(x, count), count=count)``
  * - :func:`pyspark.sql.functions.array_sort`
    - ``df.col.map(lambda x: np.sort(x))``
  * - :func:`pyspark.sql.functions.array_union`
    - ``df[['col1', 'col2']].apply(lambda x: np.union1d(x[0], x[1]), axis=1)``
  * - :func:`pyspark.sql.functions.array_overlap`
    - ``df[['A', 'B']].apply(lambda x: len(np.intersect1d(x[0], x[1])) > 0, axis=1)``
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
  * - :func:`pyspark.sql.functions.asinh`
    - ``np.arcsinh(df.col)``
  * - :func:`pyspark.sql.functions.atan`
    - ``np.arctan(df.col)``
  * - :func:`pyspark.sql.functions.atanh`
    - ``np.arctanh(df.col)``
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
  * - :func:`pyspark.sql.functions.corr`
    - ``df[['col1', 'col2']].corr(method = 'pearson')``
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
  * - :func:`pyspark.sql.functions.filter`
    - ``df.filter()``
  * - :func:`pyspark.sql.functions.floor`
    - ``np.floor(df.col)``
  * - :func:`pyspark.sql.functions.format_number`
    - ``df.col.apply(lambda x,d : ("{:,." + str(d) + "f}").format(np.round(x, d)), d=d)``
  * - :func:`pyspark.sql.functions.format_string`
    - ``df.col.apply(lambda x, format_str : format_str.format(x), format_str=format_str)``
  * - :func:`pyspark.sql.functions.from_unixtime`
    - ``df.col.map(lambda x: pd.Timestamp(x, 's')).dt.strftime(format_str)``
  * - :func:`pyspark.sql.functions.greatest`
    - ``df[['col1', 'col2']].apply(lambda x: np.nanmax(x), axis=1)``
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
    - | # If the type is uint64 ``np.left_shift(df.col.astype(np.int64), numbits).astype(np.uint64))``
      | # Other integer types: ``np.left_shift(df.col, numbits)``
  * - :func:`pyspark.sql.functions.shiftRight`
    - | # If the type is uint64 use ``shiftRightUnsigned``
      | # Other integer types: ``np.right_shift(df.col, numbits)``
  * - :func:`pyspark.sql.functions.shiftRightUnsigned`
    - | ``def shiftRightUnsigned(col, num_bits):``
      |   ``bits_minus_1 = max((num_bits - 1), 0)``
      |   ``mask_bits = (np.int64(1) << bits_minus_1) - 1``
      |   ``mask = ~(mask_bits << (63 - bits_minus_1))``
      |   ``return np.right_shift(col.astype(np.int64), num_bits) & mask).astype(np.uint64)``
      | ``shiftRightUnsigned(df.col, numbits)``
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
  * - :func:`pyspark.sql.functions.timestamp_seconds`
    - ``pd.to_datetime("now")`` 
  * - :func:`pyspark.sql.functions.to_date`
    - ``df.col.apply(lambda x, format_str: pd.to_datetime(x, format=format_str).date(), format_str=format_str)`` 
  * - :func:`pyspark.sql.functions.to_timestamp`
    - ``df.A.apply(lambda x, format_str: pd.to_datetime(x, format=format_str), format_str=format_str)`` 
  * - :func:`pyspark.sql.functions.translate`
    - ``df.col.str.split("").apply(lambda x: "".join(pd.Series(x).replace(to_replace, values).tolist()), to_replace=to_replace, values=values)``
  * - :func:`pyspark.sql.functions.trim`
    - ``df.col.str.strip()``
  * - :func:`pyspark.sql.functions.trunc`
    - | ``def f(date, trunc_str):``
      |     ``if trunc_str == 'year':``
      |         ``return pd.Timestamp(year=date.year, month=1, day=1)``
      |     ``if trunc_str == 'month':``
      |         ``return pd.Timestamp(year=date.year, month=date.month, day=1)``
      | ``df.A.apply(f, trunc_str=trunc_str)``
  * - :func:`pyspark.sql.functions.udf`
    - ``df.apply`` or ``df.col.map`` 
  * - :func:`pyspark.sql.functions.unix_timestamp`
    - ``df.col.apply(lambda x, format_str: (pd.to_datetime(x, format=format_str) - pd.Timestamp("1970-01-01")).total_seconds(), format_str=format_str)`` 
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
