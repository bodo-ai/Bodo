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


## Running TPCH Benchmarks

### Software versions

Below are the version of the software used:

| Package      | Version      |
|----------------|----------------|
| bodo   | 2025.12   |
| bodosdk | |
| polars | |
| pandas | |
| duckdb | |
| dask   |  |
| dask-cloudprovider  | |
| PySpark  | |
<!-- TODO: Daft -->
<!-- TODO: Modin -->

<!-- You can install these packages into a single environment using the provided `requirements.txt`. -->

### Single Node

<!-- TODO: describe final cluster settings -->

Single-node only libraries include Polars, DuckDB, Pandas. In addition to these we also Dask, PySpark and Bodo, which are distributed engine that can be run on a single node as well. Single Node implementations can be found in the `pds-benchmark/queries` folder, which was copied from the [Polars Decision Support Benchmark](https://github.com/pola-rs/polars-benchmark) and extended to include Bodo.

To reproduce the results for this benchmark, start by cd'ing into the benchmark folder and running
To run a specific query from a specific implementation, you can use:

``` shell
SCALE_FACTOR=SF PATH_DATA_FOLDER=/path/to/you/tpch/data python -m queries.<IMPL>.q<Q_NUM>
```
Note that this should be run from the `pds-benchmark/` directory. To run all queries from a specific implementation, you can use:
``` shell
SCALE_FACTOR=SF PATH_DATA_FOLDER=/path/to/you/tpch/data python -m queries.<IMPL>
```
These scripts will run each query (as a separate Python process) and measures the query time, which includes reading from IO. Note that each query will be run twice and only the "hot start" time will be logged. Both the cold and hot start run will be included as part of total time.

Because Bodo is a drop-in Pandas replacement, we reuse the Bodo queries for the Pandas baseline. To run the Bodo queries with a Pandas backend run the Bodo script with `BODO_USE_PANDAS_BACKEND=1`.

Note that Bodo hangs when running back-to-back queries on Mac, to get around this you can use the `./run_bodo.sh` script, which runs each query with sleeps to prevent hangs.

### Multi-node benchmark

<!-- TODO: describe final cluster settings -->

Distributed libraries like Bodo, Dask, PySpark also have the option of being multi-node. Because of the different infrastructure requirements for running these libraries, the scripts for reproducing multi-node results can be found in separate `impl/` directories.

#### Bodo

Follow [the instructions here](https://github.com/bodo-ai/Bodo/tree/main/benchmarks/nyc_taxi#bodo) to set up a Bodo Platform account through AWS marketplace and set up access tokens. You can then run the script:

``` shell
python run_bodo.py --folder s3://path/to/data --scale_factor SF --queries 1 2 ...
```

Or omit the `queries` argument to run all queries.

#### Dask

We use [Dask Cloudprovider](https://cloudprovider.dask.org/en/latest/) to create a Dask cluster with a scheduler instance (which doesn't do any compute) and worker instances. To ensure that the local environment matches the environment running the script, we provide an `env.yml` file in the same directory:

``` shell
cd dask
conda create --file env.yml
conda activate dask_tpch
python dask_queries.py --folder s3://path/to/data --scale_factor SF --queries 1 2 ...
```

#### PySpark

We used AWS EMR to create the PySpark cluster. You will need the following additional dependencies install on your local machine:

* [**AWS CLI**](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) (Installed and configured with access keys.)
* [**Terraform**](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
* [**jq**](https://jqlang.github.io/jq/download/) (for viewing logs locally)
* [**gzip**](https://www.gnu.org/software/gzip/) (for viewing logs locally)

You can then run the terraform script:

``` shell
cd pyspark
terraform apply \
  -var="scale_factor=SF" \
  -var='queries=[1,2,...]' \
  -var="data_folder=s3://path/to/data "
```

This will run the script and write logs to an S3 bucket. You can either view the logs in the AWS console or copy them directly using the following scripts:

``` shell
./wait_for_steps.sh

aws s3 cp s3://"$(terraform output --json | jq -r '.s3_bucket_id.value')"/logs/"$(terraform output --json | jq -r '.emr_cluster_id.value')" ./emr-logs --recursive --region "$(terraform output --json | jq -r '.emr_cluster_region.value')"

# View step logs with execution time result
gzip -d ./emr-logs/steps/*/*
cat ./emr-logs/steps/*/stdout
```

