This module contains some of the code used in our blog on performance comparison of Bodo vs. Spark, Dask, and Ray. Read about our findings [here](https://bodo.ai/blog/performance-and-cost-of-bodo-vs-spark-dask-ray).

# About the Queries

We derived these queries from the TPC-H benchmarks. TPC-H is a benchmark suite for business-oriented ad-hoc queries that are used to simulate real questions and is usually used to benchmark the performance of database tools for answering them.

More information can be found [here](http://www.tpc.org/tpch/)

## Generating Data in Parquet Format

### 1. Download and Install tpch-dbgen

```
    git clone https://github.com/Bodo-inc/tpch-dbgen
    cd tpch-dbgen
    make
    cd ../
```

### 2. Generate Data

Usage

```
usage: python generate_data_pq.py [-h] --folder FOLDER [--SF N] [--validate_dataset]

    -h, --help       Show this help message and exit
    folder FOLDER: output folder name (can be local folder or S3 bucket)
    SF N: data size number in GB (Default 1)
    validate_dataset: Validate each parquet dataset with pyarrow.parquet.ParquetDataset (Default True)
```

Example:

Generate 1GB data locally:

`python generate_data_pq.py --SF 1 --folder SF1`

Generate 1TB data and upload to S3 bucket:

`python generate_data_pq.py --SF 1000 --folder s3://bucket-name/`

NOTES:

This script assumes `tpch-dbgen` is in the same directory. If you downloaded it at another location, make sure to update `tpch_dbgen_location` in the script with the new location.

- If using S3 bucket, install `s3fs` and add your AWS credentials.

## Bodo

### Installation

Follow the instructions [here](https://docs.bodo.ai/installation_and_setup/install/).

For best performance we also recommend using Intel-MPI and EFA Network Interfaces (on AWS) as described [here](https://docs.bodo.ai/installation_and_setup/recommended_cluster_config/).

### Running queries

Use

`mpiexec -n N python bodo_queries.py --folder folder_path`

```
usage: python bodo_queries.py [-h] --folder FOLDER

arguments:
  -h, --help       Show this help message and exit
  --folder FOLDER  The folder containing TPCH data

```

Example:

Run with 4 cores on a local data

`export BODO_NUM_WORKERS=4; python bodo_queries.py --folder SF1`

Run with 288 cores on S3 bucket data

`export BODO_NUM_WORKERS=288; bodo_queries.py --folder s3://bucket-name/`

## Spark

### Installation

Here, we show the instructions for using PySpark with an EMR cluster.

For other cluster configurations, please follow corresponding vendor's instructions.

Follow the steps outlined in the "Launch an Amazon EMR cluster" section of the [AWS guide](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs-launch-sample-cluster.html)

In the **Software configuration** step, select `Hadoop`, `Hive`, `JupyterEnterpriseGateway`, and `Spark`.

In the **Cluster Nodes and Instances** step, choose the same instance type for both master and workers. Don't create any task instances.

### Running queries

Attach [pyspark_notebook.ipynb](./pyspark_notebook.ipynb) to your EMR cluster following the examples in the [AWS documentation](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-managed-notebooks-create.html)
