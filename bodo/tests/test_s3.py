import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from pyarrow import fs as pafs

import bodo
from bodo.tests.utils import _get_dist_arg, cast_dt64_to_ns, check_func
from bodo.utils.testing import ensure_clean2

pytestmark = pytest.mark.s3


@pytest.mark.parquet
def test_no_use_ssl(datapath, minio_server_with_s3_envs, s3_bucket):
    _, _, address = minio_server_with_s3_envs

    def test_impl(address, fpath):
        return pd.read_parquet(
            fpath,
            storage_options={"anon": False, "endpoint_url": address, "use_ssl": False},
            dtype_backend="pyarrow",
        )

    py_output = pd.read_parquet(datapath("example.parquet"), dtype_backend="pyarrow")

    check_func(
        test_impl,
        (
            f"http://{address}/",
            f"s3://{s3_bucket}/example.parquet",
        ),
        py_output=py_output,
        convert_to_nullable_float=False,
    )


# Memory leak check is disabled because to_parquet lowers a
# constant, which has a leak
# TODO: Readd memory_leak_check
def test_partition_cols(minio_server_with_s3_envs, s3_bucket):
    """Test s3 to_parquet partition_cols."""
    for case in [0, 1]:
        bd_fname = f"s3://{s3_bucket}/bd_file.pq"
        df = pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "B": [0, 1, 2, 3, 4, 5]})
        part_cols = ["A"]
        if case == 0:
            write = lambda df: df.to_parquet(bd_fname, partition_cols=part_cols)
        else:
            write = lambda df: df.to_parquet(bd_fname + "/", partition_cols=part_cols)
        write_jit = bodo.jit(write, all_args_distributed_block=True)
        with ensure_clean2(bd_fname):
            write_jit(_get_dist_arg(df, False))
            A0_actual = bodo.jit(returns_maybe_distributed=False)(
                lambda: pd.read_parquet(f"{bd_fname}/A=0", dtype_backend="pyarrow")
            )()
            A1_actual = bodo.jit(returns_maybe_distributed=False)(
                lambda: pd.read_parquet(f"{bd_fname}/A=1", dtype_backend="pyarrow")
            )()
        A0_expected = pd.DataFrame({"B": pd.Series([0, 1, 2], dtype="Int64")})
        A1_expected = pd.DataFrame({"B": pd.Series([3, 4, 5], dtype="Int64")})
        pd.testing.assert_frame_equal(
            A0_actual, A0_expected, check_column_type=False, check_dtype=False
        )
        pd.testing.assert_frame_equal(
            A1_actual, A1_expected, check_column_type=False, check_dtype=False
        )


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [
        ("s3_bucket", "bodo-test"),
        ("s3_bucket_us_west_2", "bodo-test-2"),
    ],
)
def test_s3_csv_data1(
    minio_server_with_s3_envs, bucket_fixture, datapath, bucket_name, request
):
    """
    test s3 read_csv
    reading from s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    request.getfixturevalue(bucket_fixture)

    def test_impl(fpath):
        return pd.read_csv(
            fpath,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": float, "D": int},
            dtype_backend="pyarrow",
        )

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": int, "B": float, "C": float, "D": int},
        dtype_backend="pyarrow",
    )

    check_func(
        test_impl,
        (f"s3://{bucket_name}/csv_data1.csv",),
        py_output=py_output,
        convert_to_nullable_float=False,
    )


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [
        ("s3_bucket", "bodo-test"),
        ("s3_bucket_us_west_2", "bodo-test-2"),
    ],
)
def test_s3_csv_dir(
    minio_server_with_s3_envs, bucket_fixture, datapath, bucket_name, request
):
    """
    test s3 read_csv directory
    reading from s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    the directory name has a space character in it; we had a case where this
    used to fail, so this is a test for avoiding that regression as well.
    """
    request.getfixturevalue(bucket_fixture)

    fname_dir_multi = f"s3://{bucket_name}/example multi.csv"

    def test_impl_with_dtype(fname):
        return pd.read_csv(
            fname,
            dtype={
                "one": np.float32,
                "two": str,
                "three": "bool",
                "four": np.float32,
                "five": str,
            },
            dtype_backend="pyarrow",
        )

    py_out = pd.read_csv(datapath("example.csv"), dtype_backend="pyarrow")
    # specify dtype here because small partition of dataframe causes only
    # int values(x.0) in float columns, and causes type mismatch becasue
    # pandas infer them as int columns
    check_func(test_impl_with_dtype, (fname_dir_multi,), py_output=py_out)


def test_s3_csv_data1_compressed(minio_server_with_s3_envs, s3_bucket, datapath):
    """
    test s3 read_csv
    """

    def test_impl_gzip():
        return pd.read_csv(
            "s3://bodo-test/csv_data1.csv.gz",
            names=["A", "B", "C", "D"],
            header=None,
            dtype_backend="pyarrow",
        )

    def test_impl_bz2():
        return pd.read_csv(
            "s3://bodo-test/csv_data1.csv.bz2",
            names=["A", "B", "C", "D"],
            header=None,
            dtype_backend="pyarrow",
        )

    fname = datapath("csv_data1.csv")
    py_output = pd.read_csv(
        fname, names=["A", "B", "C", "D"], header=None, dtype_backend="pyarrow"
    )

    check_func(test_impl_gzip, (), py_output=py_output, check_dtype=False)
    check_func(test_impl_bz2, (), py_output=py_output, check_dtype=False)


def test_s3_csv_data_date1(minio_server_with_s3_envs, s3_bucket, datapath):
    """
    test s3 read_csv
    """

    def test_impl():
        return pd.read_csv(
            "s3://bodo-test/csv_data_date1.csv",
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "D": int},
            parse_dates=[2],
            dtype_backend="pyarrow",
        )

    fname = datapath("csv_data_date1.csv")
    py_output = pd.read_csv(
        fname,
        names=["A", "B", "C", "D"],
        dtype={"A": int, "B": float, "D": int},
        parse_dates=[2],
        dtype_backend="pyarrow",
    )
    check_func(test_impl, (), py_output=py_output, convert_to_nullable_float=False)


def unset_aws_vars():
    """
    We need to unset the AWS env vars so it connects to actual S3 instead of MinIO
    """
    import os

    aws_env_vars = [
        "AWS_S3_ENDPOINT",
    ]
    orig_env_vars = {}
    for v in aws_env_vars:
        if v in os.environ:
            orig_env_vars[v] = os.environ[v]
            del os.environ[v]
        else:
            orig_env_vars[v] = None
    return aws_env_vars, orig_env_vars


def reset_aws_vars(aws_env_vars, orig_env_vars):
    """
    Reset the AWS env vars to their original values
    """
    import os

    for v in aws_env_vars:
        if orig_env_vars[v] is not None:
            os.environ[v] = orig_env_vars[v]


@pytest.mark.skip(reason="BSE-3369: Data link is broken.")
def test_s3_pq_anon_public_dataset(memory_leak_check):
    """
    Test pd.read_parquet(..., storage_options={"anon": True}, dtype_backend="pyarrow")
    with a public dataset on S3.
    """

    aws_env_vars, orig_env_vars = unset_aws_vars()

    # Read from a public bucket
    def impl():
        df = pd.read_parquet(
            "s3://aws-roda-hcls-datalake/opentargets_1911/19_11_target_list/part-00000-af4c14ab-5cfb-47d9-afc0-58db3bf07129-c000.snappy.parquet",
            storage_options={"anon": True},
            dtype_backend="pyarrow",
        )
        return df

    try:
        check_func(impl, ())
    finally:
        reset_aws_vars(aws_env_vars, orig_env_vars)


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [("s3_bucket", "bodo-test"), ("s3_bucket_us_west_2", "bodo-test-2")],
)
def test_s3_pq_asof1(
    minio_server_with_s3_envs, bucket_fixture, datapath, bucket_name, request
):
    """
    test s3 read_parquet
    reading from s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    request.getfixturevalue(bucket_fixture)

    def test_impl(fpath):
        return pd.read_parquet(fpath, dtype_backend="pyarrow")

    fname = datapath("asof1.pq")
    py_output = cast_dt64_to_ns(pd.read_parquet(fname, dtype_backend="pyarrow"))
    check_func(test_impl, (f"s3://{bucket_name}/asof1.pq",), py_output=py_output)


def test_s3_pq_groupby3(minio_server_with_s3_envs, s3_bucket, datapath):
    """
    test s3 read_parquet
    """

    def test_impl():
        return pd.read_parquet("s3://bodo-test/groupby3.pq", dtype_backend="pyarrow")

    fname = datapath("groupby3.pq")
    py_output = pd.read_parquet(fname, dtype_backend="pyarrow")
    check_func(test_impl, (), py_output=py_output)


def test_s3_pq_input_file_name_col(
    minio_server_with_s3_envs, s3_bucket, datapath, memory_leak_check
):
    """
    test s3 read_parquet input_file_name_col functionality
    This is only meant to test that the input_file_name functionality
    works with S3, not the correctness itself.
    """

    def test_impl():
        return pd.read_parquet(
            "s3://bodo-test/groupby3.pq",
            _bodo_input_file_name_col="filename",
            dtype_backend="pyarrow",
        )

    fname = datapath("groupby3.pq")
    py_output = pd.read_parquet(fname, dtype_backend="pyarrow")
    py_output["filename"] = "s3://bodo-test/groupby3.pq"
    check_func(test_impl, (), py_output=py_output)


def test_s3_pq_list_files(
    minio_server_with_s3_envs, s3_bucket, datapath, memory_leak_check
):
    """
    test s3 read_parquet list of files
    """

    def test_impl():
        return pd.read_parquet(
            ["s3://bodo-test/example.parquet", "s3://bodo-test/example2.parquet"],
            dtype_backend="pyarrow",
        )

    def test_impl2(fpaths):
        return pd.read_parquet(fpaths, dtype_backend="pyarrow")

    py_output_part1 = pd.read_parquet(
        datapath("example.parquet"), dtype_backend="pyarrow"
    )
    py_output_part2 = pd.read_parquet(
        datapath("example2.parquet"), dtype_backend="pyarrow"
    )
    py_output = pd.concat([py_output_part1, py_output_part2])
    check_func(test_impl, (), py_output=py_output)
    fpaths = ["s3://bodo-test/example.parquet", "s3://bodo-test/example2.parquet"]
    check_func(test_impl2, (fpaths,), py_output=py_output)


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [("s3_bucket", "bodo-test"), ("s3_bucket_us_west_2", "bodo-test-2")],
)
def test_s3_read_json(
    minio_server_with_s3_envs, bucket_fixture, datapath, bucket_name, request
):
    """
    test read_json from s3
    reading from s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    request.getfixturevalue(bucket_fixture)
    fname_file = f"s3://{bucket_name}/example.json"
    fname_dir_single = f"s3://{bucket_name}/example_single.json"
    fname_dir_multi = f"s3://{bucket_name}/example_multi.json"

    def test_impl(fname):
        return pd.read_json(
            fname, orient="records", lines=True, dtype_backend="pyarrow"
        )

    def test_impl_with_dtype(fname):
        return pd.read_json(
            fname,
            orient="records",
            lines=True,
            dtype={
                "one": np.float32,
                "two": str,
                "three": "bool",
                "four": np.float32,
                "five": str,
            },
            dtype_backend="pyarrow",
        )

    py_out = test_impl(datapath("example.json"))
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    # specify dtype here because small partition of dataframe causes only
    # int values(x.0) in float columns, and causes type mismatch becasue
    # pandas infer them as int columns
    check_func(test_impl_with_dtype, (fname_dir_multi,), py_output=py_out)


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [1.1, np.nan, 4.2, 3.1, -1.3],
                "B": [True, False, False, True, True],
                "C": [1, 4, -5, -11, 6],
            }
        )
    ]
)
def test_df(request):
    return request.param


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [("s3_bucket", "bodo-test"), ("s3_bucket_us_west_2", "bodo-test-2")],
)
# Memory leak check is disabled because to_parquet lowers a
# constant, which has a leak
# TODO: Readd memory_leak_check
def test_s3_parquet_write_seq(
    minio_server_with_s3_envs, bucket_fixture, test_df, bucket_name, request
):
    """
    test s3 to_parquet sequentially
    writing to s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    from bodo.spawn.utils import run_rank0

    request.getfixturevalue(bucket_fixture)

    def test_write(test_df, fpath):
        test_df.to_parquet(fpath)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, f"s3://{bucket_name}/test_df_bodo_seq.pq")
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        out_df = pq.read_table(
            f"{bucket_name}/test_df_bodo_seq.pq", filesystem=fs
        ).to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


# Memory leak check is disabled because to_parquet lowers a
# constant, which has a leak
# TODO: Readd memory_leak_check
def test_s3_parquet_write_1D(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_parquet in 1D distributed
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_parquet("s3://bodo-test/test_df_bodo_1D.pq")

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        out_df = pq.read_table(
            "bodo-test/test_df_bodo_1D.pq", filesystem=fs
        ).to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


# Memory leak check is disabled because to_parquet lowers a
# constant, which has a leak
# TODO: Readd memory_leak_check
def test_s3_parquet_write_1D_var(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_parquet in 1D var
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_parquet("s3://bodo-test/test_df_bodo_1D_var.pq")

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        out_df = pq.read_table(
            "bodo-test/test_df_bodo_1D_var.pq", filesystem=fs
        ).to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [("s3_bucket", "bodo-test"), ("s3_bucket_us_west_2", "bodo-test-2")],
)
def test_s3_csv_write_seq(
    minio_server_with_s3_envs, bucket_fixture, test_df, bucket_name, request
):
    """
    test s3 to_csv sequentially
    writing to s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    from bodo.spawn.utils import run_rank0

    request.getfixturevalue(bucket_fixture)

    def test_write(test_df, fpath):
        test_df.to_csv(
            fpath,
            index=False,
            header=False,
        )

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, f"s3://{bucket_name}/test_df_bodo_seq.csv")
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(
            endpoint_override="http://localhost:9000",
            region="us-east-1" if bucket_name == "bodo-test" else "us-west-2",
        )
        file_obj = fs.open_input_file(f"{bucket_name}/test_df_bodo_seq.csv")
        out_df = csv.read_csv(
            file_obj,
            read_options=csv.ReadOptions(column_names=["A", "B", "C"]),
            convert_options=csv.ConvertOptions(
                column_types={"A": "float64", "B": "bool", "C": "int64"}
            ),
        ).to_pandas()
        return out_df

    out_df = read_table()

    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_1D(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_csv in 1D distributed
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_1D.csv", index=False, header=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/test_df_bodo_1D.csv/",
            format=ds.CsvFileFormat(
                read_options=csv.ReadOptions(column_names=["A", "B", "C"]),
                convert_options=csv.ConvertOptions(
                    column_types={"A": "float64", "B": "bool", "C": "int64"}
                ),
            ),
            filesystem=fs,
            schema=pa.schema(
                [
                    ("A", pa.float64()),
                    ("B", pa.bool_()),
                    ("C", pa.int64()),
                ]
            ),
        )

        out_df = dataset.to_table().to_pandas()

        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_1D_var(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_csv in 1D var
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_csv(
            "s3://bodo-test/test_df_bodo_1D_var.csv", index=False, header=False
        )

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/test_df_bodo_1D_var.csv/",
            format=ds.CsvFileFormat(
                read_options=csv.ReadOptions(column_names=["A", "B", "C"]),
                convert_options=csv.ConvertOptions(
                    column_types={"A": "float64", "B": "bool", "C": "int64"}
                ),
            ),
            filesystem=fs,
            schema=pa.schema(
                [
                    ("A", pa.float64()),
                    ("B", pa.bool_()),
                    ("C", pa.int64()),
                ]
            ),
        )
        out_df = dataset.to_table().to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_header_seq(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_csv with header sequentially
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_seq.csv", index=False)

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df)
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        file_obj = fs.open_input_file("bodo-test/test_df_bodo_header_seq.csv")
        out_df = csv.read_csv(file_obj)
        return out_df.to_pandas()

    out_df = read_table()

    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_header_1D(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_csv with header in 1D distributed
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_1D.csv", index=False)

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/test_df_bodo_header_1D.csv/",
            format="csv",
            filesystem=fs,
            schema=pa.schema(
                [
                    ("A", pa.float64()),
                    ("B", pa.bool_()),
                    ("C", pa.int64()),
                ]
            ),
        )
        out_df = dataset.to_table().to_pandas()
        return out_df

    out_df = read_table()

    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_header_1D_var(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_csv with header in 1D var
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_csv("s3://bodo-test/test_df_bodo_header_1D_var.csv", index=False)

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/test_df_bodo_header_1D_var.csv/",
            format="csv",
            filesystem=fs,
            schema=pa.schema(
                [
                    ("A", pa.float64()),
                    ("B", pa.bool_()),
                    ("C", pa.int64()),
                ]
            ),
        )
        out_df = dataset.to_table().to_pandas()

        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


def test_s3_csv_write_file_prefix(minio_server_with_s3_envs, s3_bucket, test_df):
    """Test S3 to_csv with unique distributed file prefix"""

    def test_write(test_df):
        test_df.to_csv(
            "s3://bodo-test/test_df_bodo_file_prefix.csv", _bodo_file_prefix="test-"
        )

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
    info = fs.get_file_info(
        pafs.FileSelector("bodo-test/test_df_bodo_file_prefix.csv/")
    )
    file_names: list[str] = [f.base_name for f in info]
    assert all(f.startswith("test-") for f in file_names)


@pytest.mark.timeout(1000)
def test_s3_json_write_file_prefix(minio_server_with_s3_envs, s3_bucket, test_df):
    """Test S3 to_json with unique distributed file prefix"""

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/test_df_bodo_file_prefix.json", _bodo_file_prefix="test-"
        )

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
    info = fs.get_file_info(
        pafs.FileSelector("bodo-test/test_df_bodo_file_prefix.json/")
    )
    file_names: list[str] = [f.base_name for f in info]
    assert all(f.startswith("test-") for f in file_names)


@pytest.mark.timeout(1000)
@pytest.mark.parametrize(
    "bucket_fixture,bucket_name",
    [("s3_bucket", "bodo-test"), ("s3_bucket_us_west_2", "bodo-test-2")],
)
def test_s3_json_write_records_lines_seq(
    minio_server_with_s3_envs, bucket_fixture, test_df, bucket_name, request
):
    """
    test s3 to_json(orient="records", lines=True) sequentially
    writing to s3_bucket_us_west_2 will check if the s3 auto region
    detection functionality works
    """
    from bodo.spawn.utils import run_rank0

    request.getfixturevalue(bucket_fixture)

    def test_write(test_df, fpath):
        test_df.to_json(
            fpath,
            orient="records",
            lines=True,
        )

    bodo_write = bodo.jit(test_write)
    bodo_write(test_df, f"s3://{bucket_name}/df_records_lines_seq.json")
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(
            endpoint_override="http://localhost:9000",
            region="us-east-1" if bucket_name == "bodo-test" else "us-west-2",
        )
        dataset = ds.dataset(
            "bodo-test/df_records_lines_seq.json", format="json", filesystem=fs
        )
        out_df = dataset.to_table().to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


@pytest.mark.timeout(1000)
def test_s3_json_write_records_lines_1D(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    test s3 to_json(orient="records", lines=True) in 1D distributed
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/df_records_lines_1D.json", orient="records", lines=True
        )

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/df_records_lines_1D.json", format="json", filesystem=fs
        )
        out_df = dataset.to_table().to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


@pytest.mark.timeout(1000)
def test_s3_json_write_records_lines_1D_var(
    minio_server_with_s3_envs, s3_bucket, test_df
):
    """
    test s3 to_json(orient="records", lines=True) in 1D var
    """
    from bodo.spawn.utils import run_rank0

    def test_write(test_df):
        test_df.to_json(
            "s3://bodo-test/df_records_lines_1D_var.json", orient="records", lines=True
        )

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_df, False, True))
    bodo.barrier()

    @run_rank0
    def read_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        dataset = ds.dataset(
            "bodo-test/df_records_lines_1D_var.json", format="json", filesystem=fs
        )
        out_df = dataset.to_table().to_pandas()
        return out_df

    out_df = read_table()
    pd.testing.assert_frame_equal(
        out_df, test_df, check_dtype=False, check_column_type=False
    )


@pytest.mark.df_lib
def test_s3_parquet_read(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    read_parquet
    test the parquet file we just wrote sequentially
    """
    from bodo.spawn.utils import run_rank0

    @run_rank0
    def write_table():
        table = pa.Table.from_pandas(test_df)
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        pq.write_table(table, "bodo-test/test_df_bodo_read.pq", filesystem=fs)

    write_table()

    def test_read():
        return pd.read_parquet(
            "s3://bodo-test/test_df_bodo_read.pq", dtype_backend="pyarrow"
        )

    check_func(test_read, (), py_output=test_df.convert_dtypes(dtype_backend="pyarrow"))


def test_s3_csv_read(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    read_csv
    test the csv file we just wrote sequentially
    """
    from bodo.spawn.utils import run_rank0

    @run_rank0
    def write_table():
        table = pa.Table.from_pandas(test_df)
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        fs.create_dir("bodo-test")
        out_file_object = fs.open_output_stream("bodo-test/test_df_bodo_read.csv")
        csv.write_csv(
            table,
            out_file_object,
            write_options=csv.WriteOptions(
                include_header=False,
            ),
        )

    write_table()

    def test_read():
        return pd.read_csv(
            "s3://bodo-test/test_df_bodo_read.csv",
            names=["A", "B", "C"],
            dtype={"A": float, "B": "bool", "C": int},
            dtype_backend="pyarrow",
        )

    check_func(test_read, (), py_output=test_df.convert_dtypes(dtype_backend="pyarrow"))


def test_s3_csv_read_header(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    read_csv with header and infer dtypes
    test the csv file we just wrote sequentially
    """
    from bodo.spawn.utils import run_rank0

    @run_rank0
    def write_table():
        table = pa.Table.from_pandas(test_df)
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        out_file_object = fs.open_output_stream(
            "bodo-test/test_df_bodo_read_header.csv"
        )
        csv.write_csv(
            table,
            out_file_object,
        )

    write_table()

    def test_read():
        return pd.read_csv(
            "s3://bodo-test/test_df_bodo_read_header.csv",
            dtype_backend="pyarrow",
        )

    check_func(test_read, (), py_output=test_df)


@pytest.fixture(params=[np.arange(5, dtype=np.int64)])
def test_np_arr(request):
    return request.param


def test_s3_np_tofile_seq(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    test s3 to_file
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_seq.dat")

    bodo_write = bodo.jit(test_write)
    bodo_write(test_np_arr)


def test_s3_np_tofile_1D(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    test s3 to_file in 1D distributed
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_1D.dat")

    bodo_write = bodo.jit(all_args_distributed_block=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, True))


def test_s3_np_tofile_1D_var(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    test s3 to_file in 1D Var
    """

    def test_write(test_np_arr):
        test_np_arr.tofile("s3://bodo-test/test_np_arr_bodo_1D_var.dat")

    bodo_write = bodo.jit(all_args_distributed_varlength=True)(test_write)
    bodo_write(_get_dist_arg(test_np_arr, True, True))


def test_s3_np_fromfile_seq(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote sequentially
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_seq.dat", np.int64)

    check_func(test_read, (), py_output=test_np_arr)


def test_s3_np_fromfile_seq_count_offset(
    minio_server_with_s3_envs, s3_bucket, test_np_arr
):
    """
    fromfile with count and offset
    """

    count = 2
    offset = 1

    def test_read():
        bytes_per_int64 = 8
        return np.fromfile(
            "s3://bodo-test/test_np_arr_bodo_seq.dat",
            np.int64,
            count=count,
            offset=offset * bytes_per_int64,
        )

    check_func(test_read, (), py_output=test_np_arr[offset : offset + count])


def test_s3_np_fromfile_seq_large_count(
    minio_server_with_s3_envs, s3_bucket, test_np_arr
):
    """
    fromfile with count larger than the length of the data
    test to read all the data and not throw an error
    """

    count = len(test_np_arr) + 1

    def test_read():
        return np.fromfile(
            "s3://bodo-test/test_np_arr_bodo_seq.dat", np.int64, count=count
        )

    check_func(test_read, (), py_output=test_np_arr[:count])


def test_s3_np_fromfile_seq_large_offset(
    minio_server_with_s3_envs, s3_bucket, test_np_arr
):
    """
    fromfile with offset larger than the length of the data
    this setup raises a ValueError which is expected
    """

    offset = len(test_np_arr) + 1

    def test_read():
        bytes_per_int64 = 8
        return np.fromfile(
            "s3://bodo-test/test_np_arr_bodo_seq.dat",
            np.int64,
            offset=offset * bytes_per_int64,
        )

    with pytest.raises(ValueError, match="negative dimensions not allowed"):
        bodo.jit(distributed=False)(test_read)()


def test_s3_np_fromfile_1D(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_1D.dat", np.int64)

    check_func(test_read, (), py_output=test_np_arr)


def test_s3_np_fromfile_1D_var(minio_server_with_s3_envs, s3_bucket, test_np_arr):
    """
    fromfile
    test the dat file we just wrote in 1D
    """

    def test_read():
        return np.fromfile("s3://bodo-test/test_np_arr_bodo_1D_var.dat", np.int64)

    check_func(test_read, (), py_output=test_np_arr)


@pytest.mark.timeout(1000)
def test_s3_json_read(minio_server_with_s3_envs, s3_bucket, test_df):
    """
    read_json(orient="records", lines=True)
    test the json file we just wrote sequentially
    """
    from bodo.spawn.utils import run_rank0

    @run_rank0
    def write_table():
        fs = pafs.S3FileSystem(endpoint_override="http://localhost:9000")
        with fs.open_output_stream("bodo-test/df_records_lines.json") as f:
            test_df.to_json(
                f,
                orient="records",
                lines=True,
            )

    write_table()

    def test_read():
        return pd.read_json(
            "s3://bodo-test/df_records_lines.json",
            orient="records",
            lines=True,
            dtype_backend="pyarrow",
        )

    def test_read_infer_dtype():
        return pd.read_json(
            "s3://bodo-test/df_records_lines.json",
            orient="records",
            lines=True,
            dtype={"A": float, "B": "bool", "C": int},  # type: ignore
            dtype_backend="pyarrow",
        )

    check_func(test_read, (), py_output=test_df)
    check_func(
        test_read_infer_dtype,
        (),
        py_output=test_df.convert_dtypes(dtype_backend="pyarrow"),
    )


@pytest.mark.slow
@pytest.mark.timeout(1000)
def test_s3_json_data_has_path(
    minio_server_with_s3_envs, s3_bucket, datapath, memory_leak_check
):
    """
    test s3 read_json where data includes ://path
    """

    def test_impl():
        return pd.read_json(
            "s3://bodo-test/path_example.json", lines=True, dtype_backend="pyarrow"
        )

    py_output = pd.read_json(
        datapath("path_example.json"), lines=True, dtype_backend="pyarrow"
    )
    check_func(test_impl, (), py_output=py_output)


@pytest.mark.skip("DeltaTable doesn't seem to support custom S3 endpoints")
def test_read_parquet_from_s3_deltalake(minio_server_with_s3_envs, s3_bucket):
    """
    DeltaTable doesn't seem to support custom S3 endpoints, so we can't test
    using MinIO on CI for now.
    Run the test manually:

    import bodo
    import pandas as pd
    import os

    # 427 account
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIA...."
    os.environ["AWS_SECRET_ACCESS_KEY"] = "OOibMP..."

    def read_data(f):
        df = pd.read_parquet(f, dtype_backend="pyarrow")
        return df

    bodo_read_data = bodo.jit(distributed=["df"])(read_data)

    print(bodo_read_data("s3://deltalake-sample/simple_table"))
    ## Expected output:
    #       id
    #    0   5
    #    1   7
    #    2   9

    print(bodo_read_data("s3://deltalake-sample/example_deltalake"))
    ## Expected output:
    #       value
    #    0      1
    #    1      1
    #    2      2
    #    3      3
    #    4      2
    #    5      3
    """

    def impl():
        df = pd.read_parquet(
            "s3://bodo-test/example_deltalake", dtype_backend="pyarrow"
        )
        return df

    py_output = pd.DataFrame({"value": [1, 1, 2, 3, 2, 3]})
    check_func(impl, (), py_output=py_output, check_dtype=False)


@pytest.mark.timeout(1000)
def test_read_parquet_glob_s3(
    minio_server_with_s3_envs, s3_bucket, datapath, memory_leak_check
):
    def test_impl(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        return df

    filename = "s3://bodo-test/int_nulls_multi.pq"
    pyout = pd.read_parquet(datapath("int_nulls_multi.pq"), dtype_backend="pyarrow")
    # add glob patterns (only for Bodo, pandas doesn't support it)
    glob_pattern_1 = filename + "/part*.parquet"
    check_func(test_impl, (glob_pattern_1,), py_output=pyout, check_dtype=False)
    glob_pattern_2 = filename + "/part*-3af07a60-*ab59*.parquet"
    check_func(test_impl, (glob_pattern_2,), py_output=pyout, check_dtype=False)


@pytest.mark.timeout(1000)
def test_read_parquet_trailing_sep_s3(
    minio_server_with_s3_envs, s3_bucket, datapath, memory_leak_check
):
    def test_impl():
        df = pd.read_parquet(
            "s3://bodo-test/int_nulls_multi.pq/", dtype_backend="pyarrow"
        )
        return df

    pyout = pd.read_parquet(datapath("int_nulls_multi.pq"), dtype_backend="pyarrow")
    check_func(test_impl, (), py_output=pyout, check_dtype=False)


@pytest.mark.slow
@pytest.mark.timeout(1000)
def test_s3_csv_anon_public_dataset(memory_leak_check):
    """
    Test pd.read_csv(..., storage_options={"anon": True})
    with a public dataset on S3.
    """
    aws_env_vars, orig_env_vars = unset_aws_vars()

    # Read from a public bucket
    def impl():
        df = pd.read_csv(
            "s3://databrew-public-datasets-us-east-1/resolution.csv",
            storage_options={"anon": True},
            dtype_backend="pyarrow",
        )
        return df

    try:
        check_func(impl, ())
    finally:
        reset_aws_vars(aws_env_vars, orig_env_vars)


@pytest.mark.slow
@pytest.mark.timeout(1000)
def test_s3_json_anon_public_dataset(memory_leak_check):
    """
    Test pd.read_json(..., storage_options={"anon": True})
    with a public dataset on S3.
    """
    aws_env_vars, orig_env_vars = unset_aws_vars()

    # Read from a public bucket
    def impl():
        df = pd.read_json(
            "s3://awsglue-datasets/examples/us-legislators/all/memberships.json",
            lines=True,
            storage_options={"anon": True},
            dtype_backend="pyarrow",
        )
        # returning subset of column only (there's 'nan' vs. nan issue)
        return df[["area_id", "on_behalf_of_id", "organization_id", "role"]]

    try:
        check_func(impl, ())
    finally:
        reset_aws_vars(aws_env_vars, orig_env_vars)


@pytest.mark.timeout(1000)
def test_read_parquet_invalid_list_of_files(
    minio_server_with_s3_envs, s3_bucket, datapath
):
    from bodo.utils.typing import BodoError

    def test_impl(fnames):
        df = pd.read_parquet(fnames, dtype_backend="pyarrow")
        return df

    with pytest.raises(
        BodoError,
        match=r"Make sure the list/glob passed to read_parquet\(\) only contains paths to files \(no directories\)",
    ):
        fnames = ["s3://bodo-test/groupby3.pq", "s3://bodo-test/int_nulls_multi.pq"]
        bodo.jit(test_impl)(fnames)

    with pytest.raises(
        BodoError,
        match=r"Make sure the list/glob passed to read_parquet\(\) only contains paths to files \(no directories\)",
    ):
        fnames = ["s3://bodo-test/int_nulls_multi.pq", "s3://bodo-test/groupby3.pq"]
        bodo.jit(test_impl)(fnames)

    with pytest.raises(
        BodoError,
        match=r"Make sure the list/glob passed to read_parquet\(\) only contains paths to files \(no directories\)",
    ):
        fnames = [
            "s3://bodo-test/test_df_bodo_1D.pq",
            "s3://bodo-test/int_nulls_multi.pq",
        ]
        bodo.jit(test_impl)(fnames)
