# Bodo TPCH GPU Benchmark

This directory contains scripts for running a TPCH-style benchmark on GPU-backed DataFrame libraries (Bodo, Cudf, Dask, Polars). Currently only TPCH Q5 is implemented, but eventually all TPCH queries will be included here.

## Setup Instructions

### Environment Setup

To run the benchmark with Bodo, you will need to create a development environment and build Bodo from source, which can be done using [pixi](https://pixi.prefix.dev/latest/installation/):
``` shell
pixi shell -e default-cuda
pixi run build-bodo-cudf
```

To run the benchmark on other libraries such as Pandas-CuDF, Dask, and Polars, a conda environment is provided in `env.yml`:
``` shell
conda env create -f env.yml
conda activate rapids-26.02
```

### Bodo

The following command can be used to run Bodo including a warmup run (i.e. run once untimed first):

``` shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output
```

To see first-run performance, exclude the `--warmup` argument.
By default, all scripts will run on a single GPU unless the `--n_workers` argument is passed:

```shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output --n_workers 4
```

### CuDF

The Bodo script can also be used to benchmark CuDF Pandas using the `--library` argument (defaults to `"bodo"` if not specified):

``` shell
python run_bodo.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output --library cudf
```

Note that this script will run out of GPU memory for scale factors >1.

### Polars

The following command will run Polars using the default GPU engine:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output
```

Polars can also use Dask for it's execution engine by passing `--engine dask`:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output --engine dask
```

The default execution engine is limited to a single GPU, but if the Dask engine is specified, you can run on multiple GPUs using the `--n_workers` argument:

``` shell
python run_polars.py --root s3://bodo-example-data/tpch/SF1000 --warmup --print_output --engine dask --n_workers 4
```
