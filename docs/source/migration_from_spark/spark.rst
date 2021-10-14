
.. _sparkexamples:

Spark Examples
================

Bodo offers simplicity and maintainability of Python codes while unlocking orders of magnitude
performance improvement. Spark APIs are usually equivalent to simpler Python/Pandas APIs,
which are automatically parallelized by Bodo. This section aims to assist spark users with
their transition to Bodo. Here, we show the most common data wrangling methods in PySpark and
Pandas through brief code examples. We used the COVID-19 World Vaccination Progress dataset
that can be downloaded from `Kaggle <https://www.kaggle.com/gpreda/covid-world-vaccination-progress?select=country_vaccinations.csv>`_.
If you want to execute the code as shown below, make sure that you have `Bodo <https://docs.bodo.ai/latest/source/install.html>`_ installed. Here is a list of examples:

- :ref:`Environment Setup`
- :ref:`Load Data`
- :ref:`Display the Schema of the DataFrame`
- :ref:`Change Data Types of the DataFrame`
- :ref:`Display the Head of the DataFrame`
- :ref:`Select Columns from the DataFrame`
- :ref:`Show the Statistics of the DataFrame`
- :ref:`Drop Duplicate Values`
- :ref:`Missing Values`
- :ref:`DateTime Manipulation`
- :ref:`Filter Data Based on Conditions`
- :ref:`Aggregation Functions`
- :ref:`Sort Data`
- :ref:`Rename Columns`
- :ref:`Create New Columns`
- :ref:`User-Defined Functions`
- :ref:`Create a DataFrame`
- :ref:`Export the Data`


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

    data = spark.read.csv('country_vaccinations_by_manufacturer.csv', header = True)


.. _Display the Schema of the DataFrame:

Display the Schema of the DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Bodo::

    @bodo.jit(distributed = ['df'])
    def schema(df):
        print(df.dtypes)

    schema(df)

With PySpark::

    print(data.printSchema())


.. _Change Data Types of the DataFrame:

Change Data Types of the DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
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

    data = spark.read.csv('country_vaccinations_by_manufacturer.csv', header = True,
                      schema = StructType(fields = new_schema))
    data.printSchema()


.. _Display the Head of the DataFrame:

Display the Head of the DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

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


