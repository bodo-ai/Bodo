.. _file_io:

Scalable Data I/O
=================

..
    I think it's better if below paragraph says "Bodo provides parallel IO for all of its supported formats"

Efficient parallel data processing requires data I/O to be parallelized
effectively as well. Bodo provides parallel file I/O for many different
formats such as `Parquet <http://parquet.apache.org>`__, CSV, JSON,
Numpy binaries, `HDF5 <http://www.h5py.org>`__ and SQL databases. This
diagram demonstrates how chunks of data are partitioned among parallel
execution engines by Bodo.

..
    Figure should say dataset instead of file (dataset can consist of multiple files,
    and is the common case with big data). Also I think we could have a better figure, maybe showing the MPI
    ranks/workers

.. figure:: ../img/file-read.jpg
   :align: center
   :alt: Bodo reads dataset chunks in parallel

   Bodo reads dataset chunks in parallel


Bodo automatically parallelizes I/O for any number of cores and cluster size
without any additional API layers.

Supported formats
-----------------

Currently, Bodo supports I/O for `Parquet <http://parquet.apache.org/>`_,
CSV, SQL, JSON, `HDF5 <http://www.h5py.org/>`_ , and Numpy binaries formats.
It can read these formats from multiple filesystems, including S3, HDFS and Azure Data Lake (ADLS)
(see `File Systems`_ below for more information).

Also see :ref:`Supported pandas Operations <pandas>` for supported arguments of I/O functions.

.. _parquet-section:

Parquet
~~~~~~~

Parquet is a commonly used file format in analytics due to its efficient
columnar storage. Bodo supports the standard pandas API for reading
Parquet: ``pd.read_parquet(path)``, where path can be a parquet file or a directory with multiple parquet files
(all are part of the same dataframe)::

    import pandas as pd
    import bodo

    @bodo.jit
    def write_pq(df):
        df.to_parquet('example.pq')

    @bodo.jit
    def read_pq():
        df = pd.read_parquet('example.pq')
        return df

``to_parquet(name)`` with distributed data writes to a folder called ``name``.
Each process writes one file into the folder, but if the data is not distributed,
``to_parquet(name)`` writes to a single file called ``name``::

    df = pd.DataFrame({'A': range(10)})

    @bodo.jit
    def example1_pq(df):
        df.to_parquet('example1.pq')

    @bodo.jit(distributed={'df'})
    def example2_pq(df):
        df.to_parquet('example2.pq')

    if bodo.get_rank() == 0:
        example1_pq(df)
    example2_pq(df)

Run the code above with 4 processors::

    $ mpiexec -n 4 python example_pq.py

``example1_pq(df)`` writes 1 single file, and ``example2_pq(df)`` writes a folder containing 4 parquet files::

    .
    ├── example1.pq
    ├── example2.pq
    │   ├── part-00.parquet
    │   ├── part-01.parquet
    │   ├── part-02.parquet
    │   └── part-03.parquet

See :ref:`read_parquet() <pandas-f-in>`, :ref:`to_parquet() <pandas-f-out>` for supported arguments.


Filter pushdown
***************

Bodo can detect filters used by the code and optimize the ``read_parquet`` call by pushing the filters
down to the storage layer, so that only the rows required by the program are read. This can significantly
speed up I/O in many cases and will reduce the program's memory footprint, sometimes substantially.

.. comment:: Should we mention removal of unused columns somewhere? That is not filter pushdown but it's similar.

For example, suppose we have a large dataset that spans many years and we only need to read data
for a particular year.
With pandas, we might perform a query on the year 2021 like this::

    @bodo.jit
    def query():
        df = pd.read_parquet("s3://my-bucket/data.pq")
        df = df[(df["year"] == 2021)]
        return df.groupby("customer_key")["revenue"].max()

When compiling the above, Bodo detects the ``df[(df["year"] == 2021)]`` filter and optimizes
the ``read_parquet`` call so that it only reads data for year 2021 from S3. Because the data
will have already been filtered after reading, Bodo removes the filter operation during compilation.
Note that this requires code transformation and optimization and is something
that pandas cannot do. Bodo automatically infers which filters can be pushed down.

.. comment:: A reference for hive-partitioning below might be good, but I didn't see a good one.

If your dataset is *hive-partitioned* and partition columns appear in filter expressions,
only the files that contain relevant data are read, and the rest are discarded based on their
path. For example, if ``year`` is a partition column above and we have a dataset::

    .
    └── data.pq/
        │   ...
        ├───year=2020/
        │   ├── part-00.parquet
        │   └── part-01.parquet
        └───year=2021/
            ├── part-02.parquet
            └── part-03.parquet

Bodo will only read the files in the ``year=2021`` directory.

For non-partition columns, Bodo may discard files entirely just by looking at their parquet metadata (depending
on the filters and statistics contained in the metadata) or filter the rows during read.

.. note::
    Filter pushdown can be a very significant optimization. Please refer to the :ref:`inlining`
    section to make sure these optimizations are applied in your program.

Exploring Large Data Without Full Read
**************************************

Exploring large datasets often requires seeing its shape and a sample of the data.
Bodo is able to provide this information quickly without loading the full Parquet dataset,
which means there is no need for a large cluster with a lot of memory. For example::

    @bodo.jit
    def head_only_read():
        df = pd.read_parquet("example.pq")
        print(df.shape)
        print(df.head())

In this example, Bodo provides the shape information for the full dataset in ``df.shape``,
but only loads the first few rows that are necessary for ``df.head()``.

.. _csv-section:

CSV
~~~


CSV is a common text format for data exchange. Bodo supports most of the
standard pandas API to read CSV files::

    import pandas as pd
    import bodo

    @bodo.jit
    def write_csv(df):
        df.to_csv('example.csv')

    @bodo.jit
    def read_csv():
        df = pd.read_csv('example.csv')
        return df

Unlike ``read_csv`` in regular pandas, Bodo can read a directory that contains multiple partitioned CSV files as well.
All files in the folder must have the same number and datatype of columns. They can have different number of rows.

Usage::

    @bodo.jit
    def read_csv_folder():
        df = pd.read_csv("/path/to/folder/foldername")
        doSomething(df)



Use ``sep="\n"`` to read text files line by line into a single-column dataframe
(without creating separate columns, useful when text data is unstructured
or there are too many columns to read efficiently)::

    @bodo.jit
    def read_test():
        df = pd.read_csv("example.csv", sep="\n", names=["value"], dtype={"value": "str"})
        return df

.. note::


   Bodo uses nullable integer types of pandas to ensure type stability (see :ref:`Integer NA issue in pandas <integer-na-issue-pandas>` for more details).
   Therefore, data types must be specified explicitly for accurate performance comparisons of Bodo and pandas for ``read_csv``.


``to_csv(name)`` has different behaviors for different file systems:

    1. POSIX file systems: always writes to a single file,
    regardless of the number of processes and whether the data is distributed, but writing is still done in parallel when more than 1 processor is used::

            df = pd.DataFrame({'A': np.arange(n)})

            @bodo.jit
            def example1_csv(df):
                df.to_csv('example1.csv')

            @bodo.jit(distributed={'df'})
            def example2_csv(df):
                df.to_csv('example2.csv')

            if bodo.get_rank() == 0:
                example1_csv(df)
            example2_csv(df)

    Run the code above with 4 processors::

            $ mpiexec -n 4 python example_csv.py

    each ``example1_csv(df)`` and ``example2_csv(df)`` writes to a single file::

            .
            ├── example1.csv
            ├── example2.csv

    2. `S3`_ and `HDFS`_: distributed data is written to a folder called ``name``.
    Each process writes one file into the folder, but if the data is not distributed,
    ``to_csv(name)`` writes to a single file called ``name``::

            df = pd.DataFrame({'A': np.arange(n)})

            @bodo.jit
            def example1_csv(df):
                df.to_csv('s3://bucket-name/example1.csv')

            @bodo.jit(distributed={'df'})
            def example2_csv(df):
                df.to_csv('s3://bucket-name/example2.csv')

            if bodo.get_rank() == 0:
                example1_csv(df)
            example2_csv(df)


    Run the code above with 4 processors::

            $ mpiexec -n 4 python example_csv.py

    ``example1_csv(df)`` writes 1 single file, and ``example2_csv(df)`` writes a folder containing 4 csv files::

            .
            ├── example1.csv
            ├── example2.csv
            │   ├── part-00.csv
            │   ├── part-01.csv
            │   ├── part-02.csv
            │   └── part-03.csv

See :ref:`read_csv() <pandas-f-in>`, :ref:`to_csv() <pandas-f-out>` for supported arguments.

.. _json-section:

JSON
~~~~

For JSON, the syntax is also the same as pandas::

    @bodo.jit
    def example_write_json(df, fname):
        df.to_json(fname)

    @bodo.jit
    def example_read_json_lines_format():
        df = pd.read_json('example.json', orient = 'records', lines = True)

    @bodo.jit
    def example_read_json_multi_lines():
        df = pd.read_json('example_file.json', orient = 'records', lines = False,
            dtype={"A": float, "B": "bool", "C": int})


.. note:: - The dtype argument required when reading a regular multi-line JSON file.
          - Bodo cannot read a directory containing multiple multi-line JSON files

``to_json(name)`` has different behaviors for different file systems:

    1. POSIX file systems: ``to_json(name)`` behavior depends on ``orient`` and ``lines`` arguments.
        (1) ``DataFrame.to_json(name, orient='records', lines=True)``
        (i.e. writing `JSON Lines text file format <http://jsonlines.org/>`_) always writes to a single file,
        regardless of the number of processes and whether the data is distributed,
        but writing is still done in parallel when more than 1 processor is used::

                df = pd.DataFrame({'A': np.arange(n)})

                @bodo.jit
                def example1_json(df):
                    df.to_json('example1.json', orient='records', lines=True)

                @bodo.jit(distributed={'df'})
                def example2_json(df):
                    df.to_json('example2.json', orient='records', lines=True)

                if bodo.get_rank() == 0:
                    example1_json(df)
                example2_jsons(df)

        Run the code above with 4 processors::

                $ mpiexec -n 4 python example_json.py

        each ``example1_json(df)`` and ``example2_json(df)`` writes to a single file::

                .
                ├── example1.json
                ├── example2.json

        (2) All other combinations of values for ``orient`` and ``lines`` have the same behavior as `S3`_ and `HDFS`_ explained below.

    2. `S3`_ and `HDFS`_: distributed data is written to a folder called ``name``.
    Each process writes one file into the folder, but if the data is not distributed,
    ``to_json(name)`` writes to a file called ``name``::

            df = pd.DataFrame({'A': np.arange(n)})

            @bodo.jit
            def example1_json(df):
                df.to_json('s3://bucket-name/example1.json')

            @bodo.jit(distributed={'df'})
            def example2_json(df):
                df.to_json('s3://bucket-name/example2.json')

            if bodo.get_rank() == 0:
                example1_json(df)
            example2_json(df)


    Run the code above with 4 processors::

            $ mpiexec -n 4 python example_json.py

    ``example1_json(df)`` writes 1 single file, and ``example2_json(df)`` writes a folder containing 4 json files::

            .
            ├── example1.json
            ├── example2.json
            │   ├── part-00.json
            │   ├── part-01.json
            │   ├── part-02.json
            │   └── part-03.json

See :ref:`read_json() <pandas-f-in>`, :ref:`to_json() <pandas-f-out>` for supported arguments.

.. _sql-section:

SQL
~~~

For SQL, the syntax is also the same as pandas. For reading::

    @bodo.jit
    def example_read_sql():
        df = pd.read_sql('select * from employees', 'mysql+pymysql://admin:server')

See :ref:`read_sql() <pandas-f-in>` for supported arguments.

For writing::

    @bodo.jit
    def example_write_sql(df):
        df.to_sql('table_name', 'mysql+pymysql://admin:server')

See :ref:`to_sql() <pandas-f-in>` for supported arguments.

.. note::

  ``sqlalchemy`` must be installed in order to use ``pandas.read_sql``.

.. _deltalake-section:

Delta Lake
~~~~~~~~~~

Reading parquet files from Delta Lake is supported locally, from S3, and from Azure ADLS.
  - The Delta Lake binding python packaged needs to be installed using pip:``pip install deltalake``.
  - For S3, the ``AWS_DEFAULT_REGION`` environment variable should be set to the region of the bucket hosting the Delta Lake table.
  - For ADLS, the ``AZURE_STORAGE_ACCOUNT`` and ``AZURE_STORAGE_KEY`` environment variables need to be set.

Example code for reading::

    @bodo.jit
    def example_read_deltalake():
        df = pd.read_parquet('path/to/deltalake')


.. note::


   Writing is currently not supported.

.. _numpy-binary-section:

Numpy binaries
~~~~~~~~~~~~~~

Numpy's ``fromfile`` and ``tofile`` are supported as below::

    @bodo.jit
    def example_np_io():
        A = np.fromfile("myfile.dat", np.float64)
        ...
        A.tofile("newfile.dat")

Bodo has the same behavior as Numpy for ``numpy.ndarray.tofile()``, where we always write to a single file.
However, writing distributed data to POSIX is done in parallel,
but writing to S3 & HDFS is done sequentially (due to file system limitations).

HDF5
~~~~

HDF5 is a common format in scientific computing, especially for
multi-dimensional numerical data. HDF5 can be very efficient at scale,
since it has native parallel I/O support. For HDF5, the syntax is the same as the `h5py <http://www.h5py.org/>`_ package.
For example::

    @bodo.jit
    def example_h5():
        f = h5py.File("data.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]


.. _non-constant-filepaths:

Filepaths determined at runtime
---------------------------------

When reading from a file, Bodo needs to know the types of the resulting dataframe.
If the file name is a constant string or function argument, Bodo can look at
the file at compile time and infer the types. If the the filepath is not constant,
this information must be supplied by the user. For `pd.read_csv` and `pd.read_excel`,
this information can be supplied through the `names` and `dtypes` keyword arguments::

        @bodo.jit
        def example_csv(fname1, fname2, flag)):
            if flag:
                file_name = fname1
            else:
                file_name = fname2
            return pd.read_csv(file_name, names = ["A", "B", "C"], dtype={"A": int, "B": float, "C": str})

        @bodo.jit
        def example_excel(fname1, fname2, flag)):
            if flag:
                    file_name = fname1
            else:
                file_name = fname2
            return pd.read_excel(
                file_name,
                names=["A", "B", "C", "D", "E"],
                dtype={"A": int, "B": float, "C": str, "D": str, "E": np.bool_},
            )

For the remaining pandas read functions, the existing APIs do not currently allow this information to be supplied.
Users can still provide this information by adding type information in the ``bodo.jit`` decorator, similar to `Numba's typing syntax
<http://numba.pydata.org/numba-doc/latest/reference/types.html>`_. For
example::

    @bodo.jit(locals={'df':{'one': bodo.float64[:],
                      'two': bodo.string_array_type,
                      'three': bodo.bool_[:],
                      'four': bodo.float64[:],
                      'five': bodo.string_array_type,
                      }})
    def example_df_schema(fname1, fname2, flag):
        if flag:
            file_name = fname1
        else:
            file_name = fname2
        df = pd.read_parquet(file_name)
        return df


     @bodo.jit(locals={'X': bodo.float64[:,:], 'Y': bodo.float64[:]})
     def example_h5(fname1, fname2, flag):
        if flag:
            file_name = fname1
        else:
            file_name = fname2
         f = h5py.File(file_name, "r")
         X = f['points'][:]
         Y = f['responses'][:]

For the complete list of supported types, please see the :ref:`pandas dtype section <pandas-dtype>`.
In the event that the dtypes are improperly specified, Bodo will throw a runtime error.

.. _File Systems:

File Systems
------------

.. _S3:

Amazon S3
~~~~~~~~~

Reading and writing :ref:`CSV <csv-section>`, :ref:`Parquet <parquet-section>`, :ref:`JSON <json-section>`, and :ref:`Numpy binary <numpy-binary-section>` files from and to Amazon S3 is supported.

The ``fsspec`` package must be available, and the file path should start with :code:`s3://`::

    @bodo.jit
    def example_s3_parquet():
        df = pd.read_parquet('s3://bucket-name/file_name.parquet')

These environment variables are used for File I/O with S3 credentials:
  - ``AWS_ACCESS_KEY_ID``
  - ``AWS_SECRET_ACCESS_KEY``
  - ``AWS_DEFAULT_REGION``: default as ``us-east-1``
  - ``AWS_S3_ENDPOINT``: specify custom host name, default as AWS endpoint(``s3.amazonaws.com``)

Connecting to S3 endpoints through a proxy is supported. The proxy URI can be provided by setting one of the following environment variables (listed in order of precedence):
  - ``http_proxy``
  - ``https_proxy``
  - ``HTTP_PROXY``
  - ``HTTPS_PROXY``

Bodo uses `Apache Arrow <https://arrow.apache.org/>`_ internally for read and write of data on S3.

.. _GCS:

Google Cloud Storage
~~~~~~~~~~~~~~~~~~~~
Reading and writing :ref:`Parquet <parquet-section>` files from and to Google Cloud is
supported.

The file path should start with ``gs://`` or ``gcs://``::

    @bodo.jit
    def example_gcs_parquet():
        df = pd.read_parquet('gcs://bucket-name/file_name.parquet')

These environment variables are used for File I/O with GCS credentials:
  - ``GOOGLE_APPLICATION_CREDENTIALS``

Details for ``GOOGLE_APPLICATION_CREDENTIALS`` can be seen in the Google docs `here
<https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable>`_.

Bodo uses the fsspec-based `gcsfs <https://gcsfs.readthedocs.io/en/latest/>`_ library
internally for read and write of data on GCS.

.. _HDFS:

Hadoop Distributed File System (HDFS) and Azure Data Lake Storage (ADLS) Gen2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reading and writing :ref:`CSV <csv-section>`, :ref:`Parquet <parquet-section>`, :ref:`JSON <json-section>`, and :ref:`Numpy binary <numpy-binary-section>` files from and to Hadoop Distributed File System (HDFS) is supported.
Note that Azure Data Lake Storage Gen2 can be accessed through HDFS.

The ``openjdk`` version 8 package must be available, and the file path should start with
``hdfs://`` or ``abfs[s]://``::

    @bodo.jit
    def example_hdfs_parquet():
        df = pd.read_parquet('hdfs://host:port/dir/file_name.pq')

These environment variables are used for File I/O with HDFS:
  - ``HADOOP_HOME``: the root of your installed Hadoop distribution. Often has `lib/native/libhdfs.so`.
  - ``ARROW_LIBHDFS_DIR``: location of libhdfs. Often is ``$HADOOP_HOME/lib/native``.
  - ``CLASSPATH``: must contain the Hadoop jars. You can set these using::

        export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

Bodo uses `Apache Arrow <https://arrow.apache.org/>`_ internally for read and write of data on HDFS. ``$HADOOP_HOME/etc/hadoop/hdfs-site.xml`` provides default behaviors for the HDFS client used by Bodo. Inconsistent configurations (e.g. ``dfs.replication``) could potentially cause errors in Bodo programs.


.. _db:

Databases
----------


Currently, Bodo supports most RDBMS that work with SQLAlchemy, with a corresponding driver.

.. _snowflake-section:

Snowflake
----------

To read a dataframe from a Snowflake database, users can use ``pd.read_sql`` with their Snowflake username and password:
``pd.read_sql(query,snowflake://<username>:<password>@url)``.


Prerequisites
~~~~~~~~~~~~~~

In order to be able to query Snowflake from Bodo, you will have to install the Snowflake connector. If you're using
Bodo in a conda environment:

.. code-block:: console

    $ conda install -c conda-forge snowflake-connector-python


If you've installed Bodo using pip, then you can install the Snowflake connector using pip as well:

.. code-block:: console

    $ pip install snowflake-connector-python


Usage
~~~~~~

Bodo requires the Snowflake connection string to be passed as an argument to the ``pd.read_sql`` function.
The complete code looks as follows:

.. code-block:: python3

    import bodo
    import pandas as pd

    @bodo.jit(distributed=["df"])
    def read_snowflake(db_name, table_name):
        df = pd.read_sql(
                f"SELECT * FROM {table_name}",
                f"snowflake://user:password@url/{db_name}/schema?warehouse=warehouse_name",
            )
        return df
    df = read_snowflake(db_name, temp_table_name)

We can use the ``pd.to_sql`` method to persist a dataframe to a Snowflake table:

.. code-block:: python3

    df.to_sql('<table_name>',f"snowflake://<username>:<password>@url/<db_name>/public?warehouse=XL_WH",schema="<schema>",if_exists="append",index=False)

.. note::
    - ``index=False`` is required as Snowflake does not support indexes.
    - ``if_exists=append`` is needed if the table already exists in snowflake.