# Bodo TPCH GPU Benchmark

This directory contains scripts for running a TPCH-style benchmark on GPU-backed DataFrame libraries (Bodo, Cudf, Dask, Polars). Currently only TPCH Q5 is implemented, but eventually all TPCH queries will be included here.

## Environment Setup

To run the benchmark with Bodo, you will need to create a development environment and build Bodo from source, which can be done using [pixi](https://pixi.prefix.dev/latest/installation/):
``` shell
pixi shell -e default-cuda
pixi run build-bodo-cudf
```
You will also need to set the environment variable: `OMPI_MCA_pml="ucx"` prior to running anything.

To run the benchmark on other libraries including Dask and Polars, a conda environment is provided in `env.yml`:
``` shell
conda env create -f env.yml
conda activate rapids-26.02
```

## Bodo

The following command can be used to run Bodo with a warmup run (i.e. run once untimed first):

``` shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup
```

By default, all scripts will run on a single GPU unless the `--n_workers` argument is passed, for example:

```shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup  --n_workers 4
```

will run the benchmark on 4 GPUs if possible.

## CuDF (Pandas-CuDF)

The Bodo script can also be used to benchmark Pandas-CuDF using the `--library` argument (defaults to `"bodo"` if not specified):

``` shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup --library cudf
```

Note that this script will run out of GPU memory for scale factors >1.

## Polars-CuDF

The following command will run Polars using the default GPU engine:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup
```

Polars can also use Dask for it's execution engine by passing `--engine dask`:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup  --engine dask
```

Polar's default GPU execution engine is limited to a single GPU, but if the Dask engine is specified, you can also run on multiple GPUs using the `--n_workers` argument:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup --engine dask --n_workers 4
```

## Dask-CuDF

Dask-cudf currently does not support the `date32[pyarrow]` datatype. To work around this, we provide the script: `convert_date_timestamp.py` to rewrite the dataset:

``` shell
python convert_date_timestamp.py --src_dir s3://bodo-example-data/tpch/SF1000 --dst_dir s3://your-bucket/tpch-dask/SF1000
```

Once the dataset is rewritten, you can run the benchmark using the following command:

``` shell
python run_dask.py --root s3://your-bucket/tpch-dask/SF1000 --warmup
```

You can also run on multiple GPUs using the `--n_workers` argument:

``` shell
python run_dask.py --root s3://your-bucket/tpch-dask/SF1000 --warmup --n_workers 4
```

Dask can also run on multiple nodes using [Dask CloudProvider](https://cloudprovider.dask.org/en/latest/) with the `--run_multi_node ` argument:

``` shell
python run_dask.py \
  --root s3://your-bucket/tpch-dask/SF1000 \
  --warmup \
  --n_workers 4 \
  --run_multi_node
```

This command will launch 4 `g7e.12xlarge` instances (plus one scheduler instance) in the `us-east-2` region.
You may need to pass `--subnet_id` if those instances are not available in your default subnet's availability zone.
