.. _file_io:

File I/O
===============

Bodo automatically parallelizes I/O of different nodes in a distributed setting
without any code changes.

Supported formats
-----------------

Currently, Bodo supports I/O for `Parquet <http://parquet.apache.org/>`_,
CSV, SQL, JSON, `HDF5 <http://www.h5py.org/>`_ , and Numpy binaries formats. Also see :ref:`Supported Pandas Operations <pandas>` for supported arguments.

.. _parquet-section:

Parquet
~~~~~~~

For Parquet, the syntax is the same as Pandas:
``pd.read_parquet(path)``, where path can be a parquet file or a directory with multiple parquet files 
(all are part of the same dataframe)::

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

    df = pd.DataFrame({'A': np.arange(n)})

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

.. _csv-section:

CSV
~~~

For CSV, the syntax is also the same as Pandas::

    @bodo.jit
    def write_csv(df):
        df.to_csv('example.csv')

    @bodo.jit
    def read_csv():
        df = pd.read_csv('example.csv')
        return df

Unlike Pandas' ``read_csv``, Bodo can read a directory that contains multiple partitioned CSV files as well.

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

For JSON, the syntax is also the same as Pandas::

    @bodo.jit
    def example_write_json(df, fname):
        df.to_json(fname)

    @bodo.jit
    def example_read_json_lines_format():
        df = pd.read_json('example.json', orient = 'records', lines = True)

    @bodo.jit
    def example_read_json_multi_lines():
        # dtype argument required when reading a regular multi-line JSON file
        # cannot read a directory containing multiple multi-line JSON files
        df = pd.read_json('example_file.json', orient = 'records', lines = False,
            dtype={"A": np.float, "B": "bool", "C": np.int})


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

For SQL, the syntax is also the same as Pandas. For reading::

    @bodo.jit
    def example_read_sql():
        df = pd.read_sql('select * from employees', 'mysql+pymysql://admin:server')

See :ref:`read_sql() <pandas-f-in>` for supported arguments.

For writing::

    @bodo.jit
    def example_write_sql(df):
        df.to_sql('table_name', 'mysql+pymysql://admin:server')

See :ref:`to_sql() <pandas-f-in>` for supported arguments.

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

For HDF5, the syntax is the same as the `h5py <http://www.h5py.org/>`_ package.
For example::

    @bodo.jit
    def example_h5():
        f = h5py.File("data.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]


Input array types
-----------------

Bodo needs to know the types of input arrays. If the file name is a constant
string or function argument, Bodo tries to look at the file at compile time
and recognize the types.
Otherwise, the user is responsible for providing the types similar to
`Numba's typing syntax
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


     @bodo.jit(locals={'X': bodo.float64[:,:], 'Y': bodo.float64[:]})
     def example_h5(fname1, fname2, flag):
        if flag:
            file_name = fname1
        else:
            file_name = fname2
         f = h5py.File(file_name, "r")
         X = f['points'][:]
         Y = f['responses'][:]


File Systems
------------

.. _S3:

Amazon S3
~~~~~~~~~

Reading and writing :ref:`CSV <csv-section>`, :ref:`Parquet <parquet-section>`, :ref:`JSON <json-section>`, and :ref:`Numpy binary <numpy-binary-section>` files from and to Amazon S3 is supported. 

The ``s3fs`` package must be available, and the file path should start with :code:`s3://`::

    @bodo.jit
    def example_s3_parquet():
        df = pd.read_parquet('s3://bucket-name/file_name.parquet')

These environment variables are used for File I/O with S3 credentials:
  - ``AWS_ACCESS_KEY_ID``
  - ``AWS_SECRET_ACCESS_KEY``
  - ``AWS_DEFAULT_REGION``: default as ``us-east-1``
  - ``AWS_S3_ENDPOINT``: specify custom host name, default as AWS endpoint(``s3.amazonaws.com``)

Bodo uses `Apache Arrow <https://arrow.apache.org/>`_ internally for read and write of data on S3.

.. _HDFS:

Hadoop Distributed File System (HDFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reading and writing :ref:`CSV <csv-section>`, :ref:`Parquet <parquet-section>`, :ref:`JSON <json-section>`, and :ref:`Numpy binary <numpy-binary-section>` files from and to Hadoop Distributed File System (HDFS) is supported.  

The file path should start with ``hdfs://``::

    @bodo.jit
    def example_hdfs_parquet():
        df = pd.read_parquet('hdfs://host:port/dir/file_name.pq')

These environment variables are used for File I/O with HDFS:
  - ``HADOOP_HOME``: the root of your installed Hadoop distribution. Often has `lib/native/libhdfs.so`.
  - ``ARROW_LIBHDFS_DIR``: location of libhdfs. Often is ``$HADOOP_HOME/lib/native``.
  - ``CLASSPATH``: must contain the Hadoop jars. You can set these using::

        export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

Bodo uses `Apache Arrow <https://arrow.apache.org/>`_ internally for read and write of data on HDFS. ``$HADOOP_HOME/etc/hadoop/hdfs-site.xml`` provides default behaviors for the HDFS client used by Bodo. Inconsistent configurations(e.g. ``dfs.replication``) could potentially cause errors in Bodo programs.
