.. _connectors:

Connectors
===========

This section briefly describes the implementations and known limitations of CSV, JSON, and NumPy Binaries I/O (Parquet not discussed).
This section also explains how we read and write from and to POSIX, S3, and Hadoop file systems.

``Bodo/bodo/io``
-----------------
- ``_bodo_file_reader.h``: ``FileReader``, ``SingleFileReader``, and ``DirectoryFileReader``
  used for reading CSV, and JSON files. These classes can be easily used to read other file formats that can be read bytes by bytes.
- ``_csv_json_reader.cpp``: 
	* ``LocalFileReader`` and ``LocalDirectoryFileReader`` are used to read CSV, and JSON files from POSIX file system. 
	* ``stream_reader_read()`` is the ``read()`` function called by pandas, when we pass our own file-like objects to ``pd.read_csv()`` and ``pd.read_json()``.
	* ``count_entries()`` counts the lines for CSV files and records for JSON files.
	  Both are equivalent to rows of dataframes. Each rank only reads a portion of the input.
	* ``chunk_reader()``  calls ``count_entries()``. 
	Then calculate the start offset relative to the entire input and the size of the chunk for each rank using MPI.
- ``_csv_json_writer.cpp``: used for ``df.to_csv()`` and ``df.to_json()``
- ``_fs_io.h``: file systems dependent File I/O calls
- ``_hdfs_reader.cpp``: 
	* ``HdfsFileReader``, ``HdfsDirectoryFileReader`` used for ``pd.read_json()``, ``pd.read_csv()`` from HDFS.
	* ``hdfs_get_fs()`` used for writing all file formats to HDFS
- ``_io.cpp``:
	* handles NumPy Binaries I/O for all three file systems
	* used for ``pd.read_json()``, ``pd.read_csv()``
- ``_s3_reader.cpp``
	* ``S3FileReader``, ``S3DirectoryFileReader`` used for ``pd.read_json()``, ``pd.read_csv()`` from S3. 
	* ``s3_get_fs()`` used for writing all file formats to HDFS
- ``fs_io.py``: 
	* system dependent calls used for connecting to S3 & HDFS on the python side
	* used for ``pd.read_csv()`` & ``pd.read_json()`` typing 
- ``np_io.py``: ``numpy.fromfile()`` & ``numpy.tofile()``


``DirectoryFileReader``
-----------------------

The overall idea of ``DirectoryFileReader`` treats a directory like a file. During ``DirectoryFileReader``initialization, 
it finds and sorts all files with the corresponding extension and calculates their sizes.
When seeking into ``DirectoryFileReader``, 
we initialize ``SingleFileReader`` based on the position relative to the ``DirectoryFileReader``, 
and seek into ``SingleFileReader``. 
When reading with ``DirectoryFileReader``, we read file by file until we read all bytes needed.

Reading CSV/JSON
----------------

``pd.read_csv()`` and ``pd.read_json()`` are very similar and have similar code paths:

+------------------------------------------------+------------------------------------------------+
|                      CSV                       |                      JSON                      |
+================================================+================================================+
| ``bodo/transforms/untyped_pass.py``: find dtypes depending on inference required or ``dtype``   |
| supplied already                                                                                |
+------------------------------------------------+------------------------------------------------+
| dtypes inference: read 100 rows with pandas    | dtypes inference: read 20 rows (records) with  |
| ``read_csv()``. pandas can only read from POSIX| pandas ``read_json()``. pandas can only read   |
| and S3 (using ``pyarrow``) with file names. we | from POSIX with file names. we pass file       |
| pass file handlers to pandas for S3 & HDFS.    | handlers to pandas for S3 & HDFS. dtypes       |
|                                                | not supported for multi line json formats      | 
|                                                | because we cannot read chunks of it.           |
+------------------------------------------------+------------------------------------------------+
| ``csv_ext.CsvReader()``: calls c++ functions   | ``json_ext.JsonReader()``: calls c++ functions |
+------------------------------------------------+------------------------------------------------+
| Initialize ``SingleFileReader`` or ``DirectoryFileReader`` depending on input source            | 
+------------------------------------------------+------------------------------------------------+
| ``count_entries()``: the entire file/directory is read here. The number of bytes each rank reads|
| were calculated in ``chunk_reader()`` by evenly distributed the input byte-wise. Depending on   | 
| file formats and reading specifications, different characters are considered as the end of an   | 
| entry. Each rank reads its own chunk for the file/directory, and find offests for the start of  |
| of each entry.                                                                                  |
+------------------------------------------------+------------------------------------------------+
| ``stream_reader`` returned to pandas with start offset (depending on ranks) and size to read for|
| each rank, calculated with mpi communication, so that the input is evenly distributed           | 
| entry(row in Dataframe)-wise across ranks.                                                      |
+------------------------------------------------+------------------------------------------------+
| ``pandas`` calls ``read()`` on ``stream_reader``  and the entire file/directory is read again.  | 
| This time, bytes read from each rank can form its own dataframe.                                |
+------------------------------------------------+------------------------------------------------+

The actual reading is done with ``std::ifstream`` in POSIX, and ``arrow-cpp`` in S3 & HDFS.

Reading CSV With Headers
~~~~~~~~~~~~~~~~~~~~~~~~

When reading a file: We use ``skiprows`` to skip the header row.
When reading a directory: When initializing the directory, 
we read the first (lexigraphical order) non-empty file with the right extension in the directory to
find the number of bytes of the header. When ``DirectoryFileReader`` creates ``SingleFileReader``, 
``DirectoryFileReader`` passes down ``csv_header_bytes``, so that ``SingleFileReader`` can skip the
header row in ``seek()``. Directory size is also calculated excluding headers.

JSON Line Format Objects vs. JSON Multiline Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main difference when reading the two formats are the delimiters. 
JSON Line Format Objects is just like a CSV file where ``\n`` is the delimeter, 
but for JSON Multiline Objects, ``}`` indicates the end of one record.
When reading JSON Multiline Objects, 
we need to edit the characters read from the file with ``edit_json_multiline_obj()`` 
to ensure each rank gets a complete JSON object.

Reading Numpy Binaries
----------------------

In Posix, sequential reading is done with ``fread``, 
and parallel reading is done with ``MPI_File_read_at_all``.
In S3 & HDFS, reading in ``c++`` is done similarly to CSV & JSON: ``FileReader->read()``.

Writing CSV/JSON
----------------

``pd.to_csv()`` and ``pd.to_json()`` are very similar and have similar code paths:

+------------------------------------------------+------------------------------------------------+
|                      CSV                       |                      JSON                      |
+================================================+================================================+
| ``bodo/hiframes/pd_dataframe_ext.py`` ``@overload_method``, call pandas                         |
| ``pd.to_csv("")``/``pd.to_json("")`` in `objmode` and write to files with                       |
| ``_csv_write(parallel=False)``/``_json_write(parallel=False)``. Sequential writes stop here and |
| go into c++, but parallel writes go to ``bodo/transforms/distributed_pass.py``                  |
+------------------------------------------------+------------------------------------------------+
| handles ``header``: ``_gen_csv_header_node``   |                                                |
| decides whether header is needed or not for    |                                                |
| pandas ``to_csv()`` depending on rank and      |                                                |
| whether output is a directory. If the output is|                                                |
| a single file(POSIX), only rank 0 writes the   |                                                |
| header as all other ranks just 'append' after. |                                                |
| But when output is a directory(S3 & HDFS), all |                                                |
| ranks write their own file so all ranks write  |                                                |
| their own header.                              |                                                |
+------------------------------------------------+------------------------------------------------+
| pandas ``to_csv("")``/``to_json("")`` write to a string and then we pass the string to          |
| ``csv_write``/``json_write``                                                                    |
+------------------------------------------------+------------------------------------------------+
| - POSIX: write with ``MPI``/ ``fwrite``        | - JSON Line Format to POSIX:                   |
| - S3&HDFS: write with ``arrow-cpp``            |   write with ``MPI``/ ``fwrite``               |
|                                                | - JSON Multiline Objects to POSIX:             |
|                                                |   write with ``boost-cpp``                     |
|                                                | - S3&HDFS: with ``arrow-cpp``                  |
+------------------------------------------------+------------------------------------------------+

JSON Multiline Objects cannot be written like JSON Line Format, 
because there is no simple way to combine two multiline objects into one.

Writing Numpy Binaries
----------------------
In Posix, sequential writing is done with ``fwrite``, 
and parallel writing is done with ``MPI_File_write_at_all``.
In S3 & HDFS, ``parallel_in_order_write`` does the writing. 

See its docstring in ``_fs_io.h`` for how writing is done differently for the two file systems. 
``parallel_in_order_write`` is a general in-order write function 
where more than more processors write to the same file. 

``arrow-cpp`` does not have append implemented for S3, but has it implemented for HDFS.
Here, if ``dfs.replication`` set in ``hdfs-site.xml`` is inconsistent with the number of
nodes in the cluster, an error could happen in ``parallel_in_order_write`` because appending depends on it.
