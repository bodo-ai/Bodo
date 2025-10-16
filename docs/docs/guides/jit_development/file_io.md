Scalable Data I/O {#file_io}
=================

Efficient parallel data processing requires data I/O to be parallelized
effectively as well. Bodo provides parallel file I/O for many different
formats such as [Parquet](http://parquet.apache.org), CSV, JSON, Numpy
binaries, [HDF5](http://www.h5py.org) and SQL databases. This diagram
demonstrates how chunks of data are partitioned among parallel execution
engines by Bodo.

<br>
![Bodo reads dataset chunks in parallel](img/file-read.svg#center)
<br>

Bodo automatically parallelizes I/O for any number of cores and cluster
size without any additional API layers.


I/O Workflow {#io_workflow}
------------

Make sure I/O calls for large data are inside JIT functions
to allow Bodo to parallelize I/O and divide the data across cores
automatically
(see below for supported formats and APIs).

!!! warning
    Performing I/O in regular Python (outside JIT functions)
    causes data distribution overheads when data is passed to JIT functions.
    In addition, SPMD launch mode replicates
    data on all Python processes,
    which can result in out-of-memory errors if the data is large.
    For example, a 1 GB dataframe replicated on 1000 cores consumes
    1 TB of memory.

Bodo looks at the schema of the input dataset during compilation time to infer
the datatype of the resulting dataframe.
This requires the dataset path to be available to the compiler. The path should be either a constant string value, an argument
to the JIT function, or a simple combination of the two.
For example, the following code passes the dataset path as an argument, allowing Bodo to infer the data type of `df`:

```py
import os
import pandas as pd
import bodo

data_path = os.environ["JOB_DATA_PATH"]

@bodo.jit
def f(path):
    df = pd.read_parquet(path)
    print(df.A.sum())

f(data_path)
```

Concatenating arguments and constant values also works:

```py
import os
import pandas as pd
import bodo

data_root = os.environ["JOB_DATA_ROOT"]

@bodo.jit
def f(root):
    df = pd.read_parquet(root + "/table1.pq")
    print(df.A.sum())

f(data_root)
```

In the rare case that the path should be a dynamic value inside JIT functions,
the data types have to be specified manually
(see [Specifying I/O Data Types Manually][non-constant-filepaths]).
This is error-prone and should be avoided as much as possible.


Supported Data Formats
----------------------

Currently, Bodo supports I/O for [Parquet](http://parquet.apache.org/),
CSV, SQL, JSON, [HDF5](http://www.h5py.org/), and Numpy binaries
formats. It can read these formats from multiple filesystems, including
S3, HDFS and Azure Data Lake (ADLS) (see [File Systems](#file-systems)
below for more information).
Many databases and data warehouses such as [Snowflake][snowflake-section] are supported as well.

Also see [supported pandas APIs][pandas] for supported arguments of I/O functions.

### Parquet {#parquet-section}

Parquet is a commonly used file format in analytics due to its efficient
columnar storage. Bodo supports the standard pandas API for reading
Parquet: `pd.read_parquet(path)`, where path can be a parquet file, a
directory with multiple parquet files (all are part of the same
dataframe), a glob pattern, list of files or list of glob patterns:

```py
import pandas as pd
import bodo

@bodo.jit
def write_pq(df):
    df.to_parquet("s3://bucket-name/example.pq")

@bodo.jit
def read_pq():
    df = pd.read_parquet("s3://bucket-name/example.pq")
    return df
```

!!! note
    Bodo reads datasets in parallel using multiple cores while ensuring that the number of rows read on all cores is roughly equal.
    The size and number of row groups can affect parallel read performance significantly.
    Currently, reading any number of rows in Bodo requires reading at least one row-group. To read even a single row from a parquet dataset, the entire row-group containing that row (and its corresponding metadata) needs to be read first, and then the required row is extracted from it.
    Therefore, for best parallel read performance, there must be sufficient row-groups to minimize the number of instances where multiple cores need to read from the same row group. This means there must be at least as many row groups as the number of cores, but ideally a lot more.
    At the same time, the size of the row-groups should not be too small since this can lead to overheads.
    For more details about parquet file format, refer to the [format specification](https://parquet.apache.org/docs/concepts/){target=blank}.


`to_parquet(name)` with distributed data writes to a folder called
`name`. Each process writes one file into the folder, but if the data is
not distributed, `to_parquet(name)` writes to a single file called
`name`:

```py
df = pd.DataFrame({"A": range(10)})

# Only execute on a single core
@bodo.jit(distributed=False)
def example1_pq(df):
    df.to_parquet("example1.pq")

# Execute on all cores
@bodo.jit
def example2_pq(df):
    df.to_parquet("example2.pq")

example1_pq(df)
example2_pq(df)
```

Run the code above with 4 processors:

```shell
BODO_NUM_WORKERS=4 python example_pq.py
```

`example1_pq(df)` writes 1 single file, and `example2_pq(df)` writes a
folder containing 4 parquet files:

```console
.
├── example1.pq
├── example2.pq
│   ├── part-00.parquet
│   ├── part-01.parquet
│   ├── part-02.parquet
│   └── part-03.parquet
```

See [`read_parquet()`][pandas-f-in], [`to_parquet()`][serialization-io-conversion] for supported arguments.

#### Filter Pushdown and Column Pruning
***Filter Pushdown and Column Pruning***

Bodo can detect filters used by the code and optimize the `read_parquet`
call by pushing the filters down to the storage layer, so that only the
rows required by the program are read.
In addition, Bodo only reads the columns that are used in the program,
and prunes the unused columns.
These optimizations can significantly speed up
I/O in many cases and can substantially reduce the program's memory footprint.

For example, suppose we have a large dataset with many columns that spans many years, and
we only need to read revenue data for a particular year:

```py
@bodo.jit
def query():
    df = pd.read_parquet("s3://my-bucket/data.pq")
    df = df[df["year"] == 2021]
    return df.groupby("customer_key")["revenue"].max()
```

When compiling the above, Bodo detects the `df[df["year"] == 2021]`
filter and optimizes the `read_parquet` call so that it only reads data
for year 2021 from S3.
This requires the dataframe filtering operation to be in the same JIT function
as `read_parquet`, and the dataframe variable shouldn't be used before filtering.
Bodo also makes sure only `customer_key` and `revenue` columns are read since
other columns are not used in the programs.

In general, if the dataset is *hive-partitioned* and partition columns appear in
filter expressions, only the files that contain relevant data are read,
and the rest are discarded based on their path. For example, if `year`
is a partition column above and we have a dataset:

```console
.
└── data.pq/
    │   ...
    ├───year=2020/
    │   ├── part-00.parquet
    │   └── part-01.parquet
    └───year=2021/
        ├── part-02.parquet
        └── part-03.parquet
```

Bodo will only read the files in the `year=2021` directory.

For non-partition columns, Bodo may discard files entirely just by
looking at their parquet metadata (depending on the filters and
statistics contained in the metadata) or filter the rows during read.

!!! note
    Filter pushdown is often a very important optimization
    and critical for having manageable memory footprint in big data workloads.
    Make sure filtering happens in the same JIT function
    right after dataset read (or JIT functions for I/O are inlined, see [inlining][inlining]).


#### Exploring Large Data Without Full Read
***Exploring Large Data Without Full Read***

Exploring large datasets often requires seeing its shape and a sample of
the data. Bodo is able to provide this information quickly without
loading the full Parquet dataset, which means there is no need for a
large cluster with a lot of memory. For example:

```py
@bodo.jit
def head_only_read():
    df = pd.read_parquet("s3://my-bucket/example.pq")
    print(df.shape)
    print(df.head())
```

In this example, Bodo provides the shape information for the full
dataset in `df.shape`, but only loads the first few rows that are
necessary for `df.head()`.

### CSV {#csv-section}

CSV is a common text format for data exchange. Bodo supports most of the
standard pandas API to read CSV files:

```py
import pandas as pd
import bodo

@bodo.jit
def write_csv(df):
    df.to_csv("s3://my-bucket/example.csv")

@bodo.jit
def read_csv():
    df = pd.read_csv("s3://my-bucket/example.csv")
    return df
```

Unlike `read_csv` in regular pandas, Bodo can read a directory that
contains multiple partitioned CSV files as well. All files in the folder
must have the same number and datatype of columns. They can have
different number of rows.

Usage:

```py
@bodo.jit
def read_csv_folder():
    df = pd.read_csv("s3://my-bucket/path/to/folder/foldername")
    doSomething(df)
```

Use `sep="\n"` to read text files line by line into a single-column
dataframe (without creating separate columns, useful when text data is
unstructured or there are too many columns to read efficiently):

```py
@bodo.jit
def read_test():
    df = pd.read_csv("example.csv", sep="n", names=["value"], dtype={"value": "str"})
    return df
```

!!! note
    Bodo uses nullable integer types of pandas to ensure type stability (see
    [integer NA issue in pandas][integer-na-issue-pandas] for more details). Therefore, data types must be specified
    explicitly for accurate performance comparisons of Bodo and pandas for
    `read_csv`.

`to_csv(name)` has different behaviors for different file systems:

1.  POSIX file systems: always writes to a single file, regardless of the number of processes and whether the data is
    distributed, but writing is still done in parallel when more than 1 processor is used:

    ```py
    df = pd.DataFrame({"A": np.arange(n)})

    # Only execute on a single core
    @bodo.jit(distributed=False)
    def example1_csv(df):
        df.to_csv("example1.csv")

    # Execute on all cores
    @bodo.jit
    def example2_csv(df):
        df.to_csv("example2.csv")

    example1_csv(df)
    example2_csv(df)
    ```

    Run the code above with 4 processors:

    ```shell
    BODO_NUM_WORKERS=4 python example_csv.py
    ```

    each ``example1_csv(df)`` and ``example2_csv(df)`` writes to a single file:

    ```console
    .
    ├── example1.csv
    ├── example2.csv
    ```

2.  [S3][S3] and [HDFS][HDFS]: distributed data is written to a folder called ``name``.
    Each process writes one file into the folder, but if the data is not distributed,
    ``to_csv(name)`` writes to a single file called ``name``:

    ```py
    df = pd.DataFrame({"A": np.arange(n)})

    @bodo.jit(distributed=False)
    def example1_csv(df):
        df.to_csv("s3://bucket-name/example1.csv")

    @bodo.jit
    def example2_csv(df):
        df.to_csv("s3://bucket-name/example2.csv")

    example1_csv(df)
    example2_csv(df)
    ```

    Run the code above with 4 processors:

    ```shell
    BODO_NUM_WORKERS=4 python example_csv.py
    ```

    ``example1_csv(df)`` writes 1 single file, and ``example2_csv(df)`` writes a folder containing 4 csv files:

    ```console
    .
    ├── example1.csv
    ├── example2.csv
    │   ├── part-00.csv
    │   ├── part-01.csv
    │   ├── part-02.csv
    │   └── part-03.csv
    ```

See [`read_csv()`][pandas-f-in], [`to_csv()`][serialization-io-conversion] for supported arguments.

### JSON {#json-section}

For JSON, the syntax is also the same as pandas.

Usage:

```py
@bodo.jit
def example_write_json(df, fname):
    df.to_json(fname)

@bodo.jit
def example_read_json_lines_format():
    df = pd.read_json("example.json", orient = "records", lines = True)

@bodo.jit
def example_read_json_multi_lines():
    df = pd.read_json("example_file.json", orient = "records", lines = False,
        dtype={"A": float, "B": "bool", "C": int})
```

!!! note
    -   The `dtype` argument is required when reading a regular multi-line JSON
        file.
    -   Bodo cannot read a directory containing multiple multi-line JSON
        files
    -   Bodo's default values for ``orient`` and ``lines`` are ``records`` and ``False`` respectively.

``to_json(name)`` has different behaviors for different file systems:

1. POSIX file systems: ``to_json(name)`` behavior depends on ``orient`` and ``lines`` arguments.

    1. ``DataFrame.to_json(name, orient="records", lines=True)``
        (i.e. writing [JSON Lines text file format](http://jsonlines.org/){target="blank"}) always writes to a single file,
        regardless of the number of processes and whether the data is distributed,
        but writing is still done in parallel when more than 1 processor is used:

        ```py
        df = pd.DataFrame({"A": np.arange(n)})

        # Only execute on a single core
        @bodo.jit(distributed=False)
        def example1_json(df):
            df.to_json("example1.json", orient="records", lines=True)

        # Execute on all cores
        @bodo.jit
        def example2_json(df):
            df.to_json("example2.json", orient="records", lines=True)

        example1_json(df)
        example2_jsons(df)
        ```

        Run the code above with 4 processors:

        ```shell
        BODO_NUM_WORKERS=4 python example_json.py
        ```

        each ``example1_json(df)`` and ``example2_json(df)`` writes to a single file:

        ```console
        .
        ├── example1.json
        ├── example2.json
        ```

    2. All other combinations of values for ``orient`` and ``lines`` have the same behavior as [S3][S3] and
    [HDFS][HDFS] explained below.

2.  [S3][S3] and [HDFS][HDFS] : distributed data is written to a folder called ``name``.
    Each process writes one file into the folder, but if the data is not distributed,
    ``to_json(name)`` writes to a file called ``name``:

    ```py
    df = pd.DataFrame({"A": np.arange(n)})

    # Only execute on a single core
    @bodo.jit(distributed=False)
    def example1_json(df):
        df.to_json("s3://bucket-name/example1.json")

    # Execute on all cores
    @bodo.jit
    def example2_json(df):
        df.to_json("s3://bucket-name/example2.json")

    example1_json(df)
    example2_json(df)
    ```

    Run the code above with 4 processors:

    ```shell
    BODO_NUM_WORKERS=4 python example_json.py
    ```

    ``example1_json(df)`` writes 1 single file, and ``example2_json(df)`` writes a folder containing 4 json files:

    ```console
    .
    ├── example1.json
    ├── example2.json
    │   ├── part-00.json
    │   ├── part-01.json
    │   ├── part-02.json
    │   └── part-03.json
    ```

    See [`read_json()][pandas-f-in], [`to_json()`][serialization-io-conversion] for supported arguments.

### SQL {#sql-section}

See [Databases][db] for the list of supported Relational Database Management Systems (RDBMS) with Bodo.

For SQL, the syntax is also the same as pandas. For reading:

```py
@bodo.jit
def example_read_sql():
    df = pd.read_sql("select * from employees", "mysql+pymysql://<username>:<password>@<host>/<db_name>")
```

See [`read_sql()`][pandas-f-in] for supported arguments.

For writing:

```py
@bodo.jit
def example_write_sql(df):
    df.to_sql("table_name", "mysql+pymysql://<username>:<password>@<host>/<db_name>")
```

See [`to_sql()`][pandas-f-in] for supported arguments.

!!! note
    `sqlalchemy` must be installed in order to use `pandas.read_sql`.


#### Filter Pushdown and Column Pruning
***Filter Pushdown and Column Pruning***

Similar to Parquet read, Bodo JIT compiler is able to
push down filters to the data source and prune unused columns automatically.
For example, this program reads data from a very large Snowflake table,
but only needs limited rows and columns:

```py
@bodo.jit
def filter_ex(conn, int_val):
    df = pd.read_sql("SELECT * FROM LINEITEM", conn)
    df = df[(df["l_orderkey"] > 10) & (int_val >= df["l_linenumber"])]
    result = df["l_suppkey"]
    print(result)

filter_ex(conn, 2)
```

Bodo optimizes the query passed to `read_sql` to push filters down and
prune unused columns. In this case, Bodo will replace `SELECT * FROM LINEITEM` with
the optimized version automatically:

```sql
SELECT "L_SUPPKEY" FROM (SELECT * FROM LINEITEM) as TEMP
WHERE  ( ( l_orderkey > 10 ) AND ( l_linenumber <= 2 ) )
```


### Delta Lake {#deltalake-section}

Reading parquet files from Delta Lake is supported locally, from S3, and from Azure ADLS.

-   The Delta Lake binding python packaged needs to be installed using pip:`pip install deltalake`.
-   For S3, the `AWS_DEFAULT_REGION` environment variable should be set to the region of the bucket hosting
    the Delta Lake table.
-   For ADLS, the `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_KEY` environment variables need to be set.

Example code for reading:

```py
@bodo.jit
def example_read_deltalake():
    df = pd.read_parquet("path/to/deltalake")
```

!!! note
    Writing is currently not supported.


### Numpy binaries {#numpy-binary-section}

Numpy's `fromfile` and `tofile` are supported as below:

```py
@bodo.jit
def example_np_io():
    A = np.fromfile("myfile.dat", np.float64)
    ...
    A.tofile("newfile.dat")
```

Bodo has the same behavior as Numpy for `numpy.ndarray.tofile()`, where
we always write to a single file. However, writing distributed data to
POSIX is done in parallel, but writing to S3 & HDFS is done sequentially
(due to file system limitations).

### HDF5

HDF5 is a common format in scientific computing, especially for
multi-dimensional numerical data. HDF5 can be very efficient at scale,
since it has native parallel I/O support. For HDF5, the syntax is the
same as the [h5py](http://www.h5py.org/) package. For example:

```py
@bodo.jit
def example_h5():
    f = h5py.File("data.hdf5", "r")
    X = f["points"][:]
    Y = f["responses"][:]
```

File Systems {#File Systems}
------------

### Amazon S3 {#S3}

Reading and writing [CSV][csv-section], [Parquet][parquet-section], [JSON][json-section], and
[Numpy binary][numpy-binary-section] files from and to Amazon S3 is supported.

The file path should start with `s3://`:

```py
@bodo.jit
def example_s3_parquet():
    df = pd.read_parquet("s3://bucket-name/file_name.parquet")
```

These environment variables are used for File I/O with S3 credentials:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`: default as `us-east-1`
- `AWS_S3_ENDPOINT`: specify custom host name, default as AWS endpoint(`s3.amazonaws.com`)

Connecting to S3 endpoints through a proxy is supported. The proxy URI can be provided by setting one of the following
environment variables (listed in order of precedence):

- `http_proxy`
- `https_proxy`
- `HTTP_PROXY`
- `HTTPS_PROXY`

By default, Bodo uses [Apache Arrow](https://arrow.apache.org/) internally for read
and write of data on S3. If additional `storage_options` are provided to `pd.read_parquet`
that Arrow does not support, then S3FS will be used instead. It can be optionally installed via pip or conda:

``` shell
pip install "s3fs>=2022.1.0"
```

``` shell
conda install -c conda-forge "s3fs>=2022.1.0"
```

### Azure Filesystems (ABFS or ADLS) {#Azure}

Reading and writing [Parquet][parquet-section] files from and to Azure Filesystems is 
supported.

The file path should start with `abfs://` or `abfss://`:

```py
@bodo.jit
def example_abfs_parquet():
    df = pd.read_parquet("abfs://container-name/file_name.parquet")
```

The following parameters are used for credentials:

- *Required*: The name of the storage account, either
    - Set the `AZURE_STORAGE_ACCOUNT_NAME` environment variable
    - Specify the `account_name` parameter in the `storage_options` argument of `pd.read_parquet`
    - Use the long-form URL syntax: `abfs://{CONTAINER_NAME}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/file_name.parquet`
- *Optional*: The authentication key for the storage account, either
    - Set the `AZURE_STORAGE_ACCOUNT_KEY` environment variable
    - Specify the `account_key` parameter in the `storage_options` argument of `pd.read_parquet`
- *Optional* A SAS token for the storage account, either
    - Set the `AZURE_STORAGE_SAS_TOKEN` environment variable
    - Include in the long-form URL syntax: `abfs://{CONTAINER_NAME}@{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net/{PATH}/{SAS_TOKEN}/file_name.parquet`

Bodo uses [Apache Arrow](https://arrow.apache.org/) internally for Azure IO.

### Google Cloud Storage (GCS) {#GCS}

Reading and writing [CSV][csv-section], [JSON][json-section] and [Parquet][parquet-section] files from and to Google Cloud Storage (GCS) is supported.

The file path should start with `gs://` or `gcs://`:

```py
@bodo.jit
def example_gcs_parquet():
    df = pd.read_parquet("gcs://bucket-name/file_name.parquet")
```

By default uses the process described in [here](https://google.aip.dev/auth/4110){target="blank"} to resolve credentials.
If not running on Google Cloud Platform (GCP), this may require the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to point to a JSON file containing credentials.
Details for `GOOGLE_APPLICATION_CREDENTIALS` can be seen in the Google
docs [here](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable){target="blank"}.
Locally, one may use the `gcloud` CLI tool to authenticate:

```shell
gcloud auth application-default login
```


### Hugging Face Datasets

Bodo supports reading CSV, JSON and Parquet data from Hugging Face datasets using the [huggingface_hub](https://huggingface.co/docs/huggingface_hub) library
(which can be installed using pip or Conda).
The data path should start with `hf://`. For example:

```py
@bodo.jit
def example_hf_dataset():
    return pd.read_parquet("hf://datasets/openai/gsm8k/main/train-00000-of-00001.parquet")
```


### Hadoop Distributed File System (HDFS) {#HDFS}

Reading and writing [CSV][csv-section], [Parquet][parquet-section], [JSON][json-section], and
[Numpy binary][numpy-binary-section] files from and to Hadoop Distributed File System (HDFS) is supported.

The `openjdk` version 11 package must be available, and the file path
should start with `hdfs://`:

```py
@bodo.jit
def example_hdfs_parquet():
    df = pd.read_parquet("hdfs://host:port/dir/file_name.pq")
```

These environment variables are used for File I/O with HDFS:

-   `HADOOP_HOME`: the root of your installed Hadoop distribution. Often is `lib/native/libhdfs.so`.
-   `ARROW_LIBHDFS_DIR`: location of libhdfs. Often is `$HADOOP_HOME/lib/native`.
-   `CLASSPATH`: must contain the Hadoop jars. You can set these using:
    ```shell
    export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`
    ```

Bodo uses [Apache Arrow](https://arrow.apache.org/) internally for read
and write of data on HDFS. `$HADOOP_HOME/etc/hadoop/hdfs-site.xml`
provides default behaviors for the HDFS client used by Bodo.
Inconsistent configurations (e.g. `dfs.replication`) could potentially
cause errors in Bodo programs.


#### Setting up HDFS Credentials
There are 3 supported methods for authenticating to HDFS:

- Environment Variables
- core-site.xml

##### Azure Identities
To authenticate with Azure Identites assign an identity to your Azure VM and ensure it has the 
Storage Blob Data Contributor role for the storage account to access. This is only supported on Bodo Platform.

##### core-site.xml
!!! note
    core-site.xml authentication is not supported on Bodo Platform, instead use the environment variables.

To authenticate with a core-site.xml file Hadoop Filesystem sources its credentials from the first available
`core-site.xml` file on the `CLASSPATH`. When Hadoop is set up (including
on Bodo Platform), this file is usually created at
`$HADOOP_HOME/etc/hadoop/core-site.xml` automatically.
You can edit this file and set credentials appropriately.

You can also write the core-site configuration to `bodo.HDFS_CORE_SITE_LOC`,
which is a temporary file Bodo adds to the `CLASSPATH` when it is initialized:

```py
import bodo

# Initialize the temporary directory where the core-site file
# will be written
bodo.HDFS_CORE_SITE_LOC_DIR.initialize()

# Define the core-site for your regular ADLS/HDFS read/write
# operations
CORE_SITE_SPEC = """
<configuration>
...
</configuration>
"""

# Write it to the temporary core-site file.
# Do it on one rank on every node to avoid filesystem conflicts.
if bodo.get_rank() in bodo.get_nodes_first_ranks():
    with open(bodo.HDFS_CORE_SITE_LOC, 'w') as f:
        f.write(CORE_SITE_SPEC)


@bodo.jit
def etl_job():
    df = pd.read_parquet("hdfs://.../file.pq")
    # ..
    # .. Some ETL Processing
    # ..
    df.to_parquet("hdfs://.../out_file.pq")
    return


etl_job()
```


Databases {#db}
---------

Currently, Bodo supports most RDBMS that work with SQLAlchemy, with a
corresponding driver.

Snowflake {#snowflake-section}
---------

### Prerequisites

In order to be able to query Snowflake or write a dataframe to
Snowflake from Bodo, installing the Snowflake connector is
necessary (it is installed by default on Bodo Platform).

If you have installed Bodo using pip, then you can install the Snowflake
connector using pip as well:

``` shell
pip install snowflake-connector-python
```

If you are using Bodo in a conda environment:

``` shell
conda install -c conda-forge snowflake-connector-python
```

## Reading from Snowflake

To read a dataframe from a Snowflake database, users can use
`pd.read_sql` with their Snowflake username and password:
`pd.read_sql(query, "snowflake://<username>:<password>@url")`.


### Usage

Bodo requires the Snowflake connection string to be passed as an
argument to the `pd.read_sql` function. The complete code looks as
follows:

``` py
import bodo
import pandas as pd

@bodo.jit
def read_snowflake(db_name, table_name):
    df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            f"snowflake://user:password@url/{db_name}/schema?warehouse=warehouse_name",
        )
    return df
df = read_snowflake(db_name, temp_table_name)
```
- `_bodo_read_as_dict` is a Bodo specific argument which forces
    the specified string columns to be read with dictionary-encoding. Bodo automatically loads string columns using dictionary encoding when it determines it would be beneficial based on a heuristic. Dictionary-encoding stores data in memory in an efficient manner and is most effective when the column has many repeated values. Read more about dictionary-encoded layout [here](https://arrow.apache.org/docs/format/Columnar.html#dictionary-encoded-layout){target=blank}.
    Bodo will raise a warning if the specified columns are not present in the schema or if they are not of type string.


    For example:
    ```py
    @bodo.jit
    def impl(query, conn):
        df = pd.read_sql(query, conn, _bodo_read_as_dict=["A", "B", "C"])
        return df
    ```

## Writing to Snowflake {#snowflake-write}

To write a dataframe to Snowflake using Bodo, users can use
`df.to_sql` with their Snowflake username and password:
`df.to_sql(table_name, "snowflake://<username>:<password>@url")`.


### Usage {#snowflake-write-usage}


Bodo requires the Snowflake connection string to be passed as an
argument to the `df.to_sql` function. The complete code looks as
follows:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def write_to_snowflake(df, table_name):
    df.to_sql(
        table_name,
        "snowflake://user:password@url/db_name/schema?warehouse=warehouse_name&role=role_name",
    )

write_to_snowflake(df, table_name)
```


!!! note
    - Writing Pandas Dataframe index to Snowflake is not supported. If `index` and/or `index_label`
      are provided, they will be ignored.
    - `if_exists=append` is needed if you want to append to a table that already exists in Snowflake.
    - `chunksize` argument is supported and used for writing parquet files to Snowflake stage
      in batches. It is the maximum number of rows to write to any of the intermediate parquet files.
      When not provided, Bodo will estimate and use a reasonable chunk size.
    - `dtype` and `method` arguments are not supported and will be ignored if provided.

### Required Snowflake Permissions

#### Creating a new table {#snowflake-write-perms-create}
***Creating a new table***

To create a new table, the role being used must have the `USAGE` permission
at the database level. In addition, the following permissions are
required at the Schema level:

- `USAGE`
- `CREATE TABLE`
- `CREATE STAGE`

#### Replacing an existing table
***Replacing an existing table***

To replace an existing table (i.e. when using `if_exists='replace'`), the role
must be an owner of the table, in addition to having the permissions listed in the
[create section](#snowflake-write-perms-create) (at the Database and Schema level).

#### Appending to an existing table
***Appending to an existing table***

To append to an existing table (i.e. when using `if_exists='append'`), the role
must have the `INSERT` permission at the table level, in addition to the
permissions listed in the [create section](#snowflake-write-perms-create)
(at the Database and Schema level).

#### Verifying your role's permissions in Snowflake
***Verifying your role's permissions in Snowflake***

You can check the permissions granted to your role in Snowflake
using the
[`SHOW GRANTS` command](https://docs.snowflake.com/en/sql-reference/sql/show-grants.html){target=blank}, e.g.:

```sql
show grants to role test_role
```

Alternatively, you can check the permissions at the database/schema/table level
and verify that your role has the required permissions, e.g.:

```sql
-- Check that your role has the USAGE permission on the database
show grants on database test_db
-- Check that your role has the required permissions on the schema
show grants on schema test_schema
-- Check that your role has the required permissions on the table
show grants on table test_schema."TEST_TABLE"
```

You can also use the
[Snowsight UI](https://docs.snowflake.com/en/user-guide/ui-snowsight.html){target=blank}
to check these permissions by navigating to the `Privileges` section of the
`Details` tab of the database/schema/table.


### Advanced Configuration Options

Bodo provides highly performant distributed Snowflake write capability.
This is done by writing parquet files to a Snowflake stage and then
using Snowflake's `COPY INTO` to load the data into the Snowflake table.
Based on the type of your Snowflake account (i.e. whether it is
an AWS or Azure or GCP based account) and your environment (i.e.
whether certain packages and modules are installed), Bodo will
use one of two strategies to write to Snowflake:
Direct Upload (preferred) or Put Upload.

1.  Direct Upload: In this strategy, Bodo creates a temporary stage
    and writes parquet files to it directly. Once these files are
    written, Bodo will automatically execute the `COPY INTO` command
    to copy the data into Snowflake. This is supported on
    S3 and ADLS based Snowflake stages (used by AWS and Azure based
    Snowflake accounts, respectively). Note that Bodo will drop the
    temporary stage once the data has been written. Temporary stages
    are also automatically cleaned up by Snowflake after the session ends.

2.  Put Upload: In this strategy, Bodo creates a 'named' stage, writes
    parquet files to a temporary directory locally and then uses the
    Snowflake Python Connector to upload these files to this named stage
    using the `PUT` command. Similar to the Direct Upload strategy,
    once the files have been transferred, Bodo will automatically
    execute the `COPY INTO` command to copy the data into Snowflake.
    This is used for GCS based stages (used by GCP based Snowflake
    accounts), or when the user environment doesn't have all the
    required packages and modules to use the Direct Upload strategy.

    Similar to the Direct Upload strategy, Bodo will drop the named stage
    after the data has been written to the table.

    !!! note
        In some cases, e.g. during abnormal exits, Bodo may not be able
        to drop these stages, which may require manual cleanup by the
        user. All stages created by Bodo are of the form
        "bodo_io_snowflake_{random-uuid}". You can list all stages
        created by Bodo in Snowflake by executing the
        [`SHOW STAGES` command](https://docs.snowflake.com/en/sql-reference/sql/show-stages.html){target=blank}:

        ```sql
        show stages like 'bodo_io_snowflake_%';
        ```

        and then delete them using the
        [`DROP STAGE` command](https://docs.snowflake.com/en/sql-reference/sql/drop-stage.html){target=blank}, e.g.:

        ```sql
        drop stage "bodo_io_snowflake_02ca9beb-eaf6-4957-a6ff-ff426441cd7a";
        ```

        These operations are not supported in Bodo and must be executed directly
        through the Snowflake console.

Direct Upload is the preferred strategy and used by default whenever
possible. When this is not possible, Bodo will show a warning to the
user. For optimal Snowflake ingestion performance, Bodo writes
multiple small intermediate parquet files on each rank, instead of a
single file like it does for regular parquet writes.

Users can set the environment variable `BODO_SF_WRITE_DEBUG` (to any value),
to get more details during the Snowflake write process. This includes printing
the raw result of the `COPY INTO` command execution, and printing more
detailed error messages in case of exceptions.

Bodo exposes the following configuration variables for tuning
the write process further (for advanced users only):

- `SF_WRITE_UPLOAD_USING_PUT`: This configuration variable can be set to `True`
  to force Bodo to use the Put Upload strategy. This is `False` by default
  since Direct Upload is the preferred strategy.
- `SF_WRITE_PARQUET_COMPRESSION`: This configuration variable can set to
  specify the compression method used for writing the intermediate
  Parquet files. Supported values include: `"snappy"`, `"gzip"`
  and `"zstd"`. Bodo uses `"snappy"` by default since it provided
  the best overall performance in our benchmarks, however this can vary
  based on your data.
- `SF_WRITE_PARQUET_CHUNK_SIZE`: This configuration variable can be
  used to specify the chunk size to use when writing the dataframe to
  the intermediate Parquet files. This is measured by the uncompressed
  memory usage of the dataframe (in bytes). Note that when provided,
  `df.to_sql`'s `chunksize` parameter (see description in note in
  the [Usage section](#snowflake-write-usage)) overrides this value.


These can be set as follows:
```py
import bodo
import bodo.io.snowflake

bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = False
bodo.io.snowflake.SF_WRITE_PARQUET_COMPRESSION = "gzip"
...
```

MySQL
-----



### Prerequisites

In addition to ``sqlalchemy``, installing ``pymysql`` is required. If you have installed Bodo using pip:

```shell
pip install pymysql
```

If you are using Bodo in a conda environment:

```shell
conda install pymysql -c conda-forge
```


### Usage

Reading result of a SQL query in a dataframe:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def read_mysql(table_name, conn):
    df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            conn
        )
    return df


table_name = "test_table"
conn = f"mysql+pymysql://{username}:{password}@{host}/{db_name}"
df = read_mysql(table_name, conn)
```

Writing dataframe as a table in the database:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def write_mysql(df, table_name, conn):
    df.to_sql(table, conn)


table_name = "test_table"
df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
conn = f"mysql+pymysql://{username}:{password}@{host}/{db_name}"
write_mysql(df, table_name, conn)
```

## Oracle Database

### Prerequisites

In addition to ``sqlalchemy``, install ``cx_oracle`` and Oracle instant client driver.
If you have installed Bodo using pip:

```shell
pip install cx-Oracle
```

If you are using Bodo in a conda environment:

```shell
conda install cx_oracle -c conda-forge
```

- Then, Download "Basic" or "Basic light" package matching your operating system from [here](https://www.oracle.com/database/technologies/instant-client/downloads.html){target=blank}.
- Unzip package and add it to ``LD_LIBRARY_PATH`` environment variable on Linux. For MacOS, use `init_oracle_client()` in your application to pass the Oracle Client directory name. See [Using cx_Oracle.init_oracle_client()](https://cx-oracle.readthedocs.io/en/latest/user_guide/initialization.html#usinginitoracleclient) to set the Oracle Client directory.

!!! note
    For linux ``libaio`` package is required as well.

    - pip: ``pip install libaio``
    - conda: ``conda install libaio -c conda-forge``

See [cx_oracle](https://cx-oracle.readthedocs.io/en/latest/user_guide/installation.html#cx-oracle-8-installation>){target=blank} for more information.
Alternatively, Oracle instant driver can be automatically downloaded using ``wget`` or ``curl`` commands.
Here's an example of automatic installation on a Linux OS machine.

```shell
conda install cx_oracle libaio -c conda-forge
mkdir -p /opt/oracle
cd /opt/oracle
wget https://download.oracle.com/otn_software/linux/instantclient/215000/instantclient-basic-linux.x64-21.5.0.0.0dbru.zip
unzip instantclient-basic-linux.x64-21.5.0.0.0dbru.zip
export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_5:$LD_LIBRARY_PATH
```

### Usage

Reading result of a SQL query in a dataframe:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def read_oracle(table_name, conn):
    df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            conn
        )
    return df


table_name = "test_table"
conn = f"oracle+cx_oracle://{username}:{password}@{host}/{db_name}"
df = read_oracle(table_name, conn)
```



Writing dataframe as a table in the database:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def write_mysql(df, table_name, conn):
    df.to_sql(table, conn)


table_name = "test_table"
df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
conn = f"oracle+cx_oracle://{username}:{password}@{host}/{db_name}"
write_mysql(df, table_name, conn)
```

## PostgreSQL

### Prerequisites
In addition to `sqlalchemy`, install `psycopg2`.

If you have installed Bodo using pip:

```shell
pip install psycopg2
```

If you are using Bodo in a conda environment:

```shell
conda install psycopg2 -c conda-forge
```

### Usage

Reading result of a SQL query in a dataframe:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def read_postgresql(table_name, conn):
    df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            conn
        )
    return df


table_name = "test_table"
conn = f"postgresql+psycopg2://{username}:{password}@{host}/{db_name}"
df = read_postgresql(table_name, conn)
```

Writing dataframe as a table in the database:

```py
import bodo
import pandas as pd


@bodo.jit(distributed=["df"])
def write_postgresql(df, table_name, conn):
    df.to_sql(table, conn)


table_name = "test_table"
df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
conn = f"postgresql+psycopg2://{username}:{password}@{host}/{db_name}"
write_postgresql(df, table_name, conn)
```
[comment]: <> (Autorefs in [pandas], [pandas-f-in], [serialization-io-conversion], [inlining] and [integer-na-issue-pandas] will populate as those sections are added.)
[todo]: <> (Modify/remove the comment above as the [pandas], [pandas-f-in], [serialization-io-conversion], [inlining] and [integer-na-issue-pandas] sections are added.)


Specifying I/O Data Types Manually {#non-constant-filepaths}
----------------------------------

In some rase use cases, the dataset path cannot be a constant value
or a JIT function argument. In such cases, the path is determined dynamically, which does not allow automatic Bodo data type inference.
Therefore, the user has to provide the data types manually.
For example, `names` and `dtypes` keyword arguments of `pd.read_csv` and `pd.read_excel`
allow the user to specify the data types:

```py
@bodo.jit
def example_csv(fname1, fname2, flag):
    if flag:
        file_name = fname1
    else:
        file_name = fname2
    return pd.read_csv(file_name, names = ["A", "B", "C"], dtype={"A": int, "B": float, "C": str})
```

For other pandas read functions, the existing APIs do not
currently allow this information to be provided.
Users can still provide
typing information in the `bodo.jit` decorator,
similar to [Numba's typing syntax](http://numba.pydata.org/numba-doc/latest/reference/types.html){target="blank"}.
For example:

```py
@bodo.jit(locals={"df":{"one": bodo.types.float64[:],
                  "two": bodo.types.string_array_type,
                  "three": bodo.types.bool_[:],
                  "four": bodo.types.float64[:],
                  "five": bodo.types.string_array_type,
                  }})
def example_df_schema(fname1, fname2, flag):
    if flag:
        file_name = fname1
    else:
        file_name = fname2
    df = pd.read_parquet(file_name)
    return df


 @bodo.jit(locals={"X": bodo.types.float64[:,:], "Y": bodo.types.float64[:]})
 def example_h5(fname1, fname2, flag):
    if flag:
        file_name = fname1
    else:
        file_name = fname2
     f = h5py.File(file_name, "r")
     X = f["points"][:]
     Y = f["responses"][:]
```

For the complete list of supported types, please see the [pandas dtype section][pandas-dtype].
In the event that the dtypes are improperly specified, Bodo will throw a runtime error.

!!! warning
    Providing data types manually is error-prone and should be
    avoided as much as possible.
