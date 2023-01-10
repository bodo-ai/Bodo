# Copyright (C) 2022 Bodo Inc. All rights reserved.
import gc
import glob
import hashlib
import json
import operator
import os
import subprocess
from typing import Protocol

import pytest
from mpi4py import MPI
from numba.core.runtime import rtsys

import bodo
import bodo.utils.allocation_tracking


# Similar to Pandas
class DataPath(Protocol):
    def __call__(self, *args: str, check_exists: bool = True) -> str:
        ...


@pytest.fixture(scope="session")
def datapath() -> DataPath:
    """Get the path to a test data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``bodo/tests/data``

    Returns
    -------
    path : path including ``bodo/tests/data``.

    Raises
    ------
    ValueError
        If the path doesn't exist.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

    def deco(*args, check_exists=True):
        path = os.path.join(BASE_PATH, *args)
        if check_exists and not os.path.exists(path):
            msg = "Could not find file {}."
            raise ValueError(msg.format(path))
        return path

    return deco


@pytest.fixture(scope="function")
def memory_leak_check():
    """
    A context manager fixture that makes sure there is no memory leak in the test.
    Equivalent to Numba's MemoryLeakMixin:
    https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/tests/support.py#L688
    """
    gc.collect()
    old = rtsys.get_allocation_stats()
    old_bodo = bodo.utils.allocation_tracking.get_allocation_stats()
    yield
    gc.collect()
    new = rtsys.get_allocation_stats()
    new_bodo = bodo.utils.allocation_tracking.get_allocation_stats()
    old_stats = [old, old_bodo]
    new_stats = [new, new_bodo]
    total_alloc = sum([m[0] for m in new_stats]) - sum([m[0] for m in old_stats])
    total_free = sum([m[1] for m in new_stats]) - sum([m[1] for m in old_stats])
    total_mi_alloc = sum([m[2] for m in new_stats]) - sum([m[2] for m in old_stats])
    total_mi_free = sum([m[3] for m in new_stats]) - sum([m[3] for m in old_stats])
    assert total_alloc == total_free
    assert total_mi_alloc == total_mi_free


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    """
    azure_1p_markers = [
        pytest.mark.bodo_1of9,
        pytest.mark.bodo_2of9,
        pytest.mark.bodo_3of9,
        pytest.mark.bodo_4of9,
        pytest.mark.bodo_5of9,
        pytest.mark.bodo_6of9,
        pytest.mark.bodo_7of9,
        pytest.mark.bodo_8of9,
        pytest.mark.bodo_9of9,
    ]
    azure_2p_markers = [
        pytest.mark.bodo_1of15,
        pytest.mark.bodo_2of15,
        pytest.mark.bodo_3of15,
        pytest.mark.bodo_4of15,
        pytest.mark.bodo_5of15,
        pytest.mark.bodo_6of15,
        pytest.mark.bodo_7of15,
        pytest.mark.bodo_8of15,
        pytest.mark.bodo_9of15,
        pytest.mark.bodo_10of15,
        pytest.mark.bodo_11of15,
        pytest.mark.bodo_12of15,
        pytest.mark.bodo_13of15,
        pytest.mark.bodo_14of15,
        pytest.mark.bodo_15of15,
    ]
    # BODO_TEST_PYTEST_MOD environment variable indicates that we only want
    # to run the tests from the given test file. In this case, we add the
    # "single_mod" mark to the tests belonging to that module. This envvar is
    # set in runtests.py, which also adds the "-m single_mod" to the pytest
    # command (thus ensuring that only those tests run)
    module_to_run = os.environ.get("BODO_TEST_PYTEST_MOD", None)
    if module_to_run is not None:
        for item in items:
            if module_to_run == item.module.__name__.split(".")[-1] + ".py":
                item.add_marker(pytest.mark.single_mod)

    for i, item in enumerate(items):
        # Divide the tests evenly so long tests don't end up in 1 group
        azure_1p_marker = azure_1p_markers[i % len(azure_1p_markers)]
        azure_2p_marker = azure_2p_markers[i % len(azure_2p_markers)]
        # All of the test_s3.py tests must be on the same rank because they
        # haven't been refactored to remove cross-test dependencies.
        testfile = item.module.__name__.split(".")[-1] + ".py"
        if "test_s3.py" in testfile:
            azure_1p_marker = azure_1p_markers[0]
            azure_2p_marker = azure_2p_markers[0]
        item.add_marker(azure_1p_marker)
        item.add_marker(azure_2p_marker)

    # Check if we should try and mark groups for AWS Codebuild
    if "NUMBER_GROUPS_SPLIT" in os.environ:
        num_groups = int(os.environ["NUMBER_GROUPS_SPLIT"])
        with open("testtiming.json", "r") as f:
            marker_groups = json.load(f)

        for item in items:
            # Gives filename + function name
            testname = item.module.__name__.split(".")[-1] + ".py" + "::" + item.name
            if testname in marker_groups:
                group_marker = marker_groups[testname]
            else:
                group_marker = group_from_hash(testname, num_groups)
            item.add_marker(getattr(pytest.mark, group_marker))


def group_from_hash(testname, num_groups):
    """
    Hash function to randomly distribute tests not found in the log.
    Keeps all s3 tests together in group 0.
    """
    if "test_s3.py" in testname:
        return "0"
    # TODO(Nick): Replace with a cheaper function.
    # Python's builtin hash fails on mpiexec -n 2 because
    # it has randomness. Instead we use a cryptographic hash,
    # but we don't need that level of support.
    hash_val = hashlib.sha1(testname.encode("utf-8")).hexdigest()
    # Hash val is a hex-string
    int_hash = int(hash_val, base=16) % num_groups
    return str(int_hash)


@pytest.fixture(scope="session")
def minio_server():
    """
    spins up minio server
    """
    # Session level environment variables used for S3 Testing.
    os.environ["AWS_ACCESS_KEY_ID"] = "bodotest1"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "bodosecret1"

    host, port = "127.0.0.1", "9000"
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    address = "{}:{}".format(host, port)

    os.environ["MINIO_ROOT_USER"] = access_key
    os.environ["MINIO_ROOT_PASSWORD"] = secret_key
    # For compatibility with older MinIO versions.
    os.environ["MINIO_ACCESS_KEY"] = access_key
    os.environ["MINIO_SECRET_KEY"] = secret_key
    os.environ["AWS_S3_ENDPOINT"] = "http://{}/".format(address)

    cwd = os.getcwd()
    args = [
        "minio",
        "--compat",
        "server",
        "--quiet",
        "--address",
        address,
        cwd + "/Data",
    ]
    proc = None

    try:
        if bodo.get_rank() == 0:
            proc = subprocess.Popen(args, env=os.environ)
    except (OSError, IOError):
        pytest.skip("`minio` command cannot be located")
    else:
        yield access_key, secret_key, address
    finally:
        if bodo.get_rank() == 0:
            if proc is not None:
                proc.kill()
            import shutil

            shutil.rmtree(cwd + "/Data")


def s3_bucket_helper(minio_server, datapath, bucket_name, region="us-east-1"):
    """
    creates a bucket with name $bucket_name in region $region and adds files to it
    """
    boto3 = pytest.importorskip("boto3")
    botocore = pytest.importorskip("botocore")

    access_key, secret_key, address = minio_server

    if bodo.get_rank() == 0:
        s3 = boto3.resource(
            "s3",
            endpoint_url="http://{}/".format(address),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=botocore.client.Config(signature_version="s3v4"),
            region_name=region,
        )
        bucket = s3.Bucket(bucket_name)
        bucket.create()
        test_s3_files = [
            ("csv_data1.csv", datapath("csv_data1.csv")),
            ("csv_data_date1.csv", datapath("csv_data_date1.csv")),
            ("asof1.pq", datapath("asof1.pq")),
            ("groupby3.pq", datapath("groupby3.pq")),
            ("example.json", datapath("example.json")),
            ("example.csv", datapath("example.csv")),
            ("example.parquet", datapath("example.parquet")),
            ("example2.parquet", datapath("example2.parquet")),
            ("path_example.json", datapath("path_example.json")),
        ]
        for s3_key, file_name in test_s3_files:
            s3.meta.client.upload_file(file_name, bucket_name, s3_key)
            if file_name.endswith("csv_data1.csv"):
                # upload compressed versions of this file too for testing
                subprocess.run(["gzip", "-k", "-f", file_name])
                subprocess.run(["bzip2", "-k", "-f", file_name])
                s3.meta.client.upload_file(
                    file_name + ".gz", "bodo-test", "csv_data1.csv.gz"
                )
                s3.meta.client.upload_file(
                    file_name + ".bz2", "bodo-test", "csv_data1.csv.bz2"
                )
                os.remove(file_name + ".gz")
                os.remove(file_name + ".bz2")

        def upload_dir(prefix, dst_dirname, extension):
            """upload all files with same given extension in a directory to s3"""
            pat = prefix + f"/*.{extension}"
            for path in glob.glob(pat):
                fname = path[len(prefix) + 1 :]
                fname = f"{dst_dirname}/{fname}"
                s3.meta.client.upload_file(path, bucket_name, fname)

        upload_dir(datapath("example_single.json"), "example_single.json", "json")
        upload_dir(datapath("example_multi.json"), "example_multi.json", "json")
        upload_dir(datapath("int_nulls_multi.pq"), "int_nulls_multi.pq", "parquet")
        upload_dir(datapath("example multi.csv"), "example multi.csv", "csv")

        path = datapath("example_deltalake")
        for root, dirs, files in os.walk(path):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.join(
                    "example_deltalake", os.path.relpath(full_path, path)
                )
                s3.meta.client.upload_file(full_path, bucket_name, rel_path)

    bodo.barrier()
    return bucket_name


@pytest.fixture(scope="session")
def s3_bucket(minio_server, datapath):
    """
    creates a bucket called bodo-test in s3 (region us-east-1) and adds files to it
    """
    return s3_bucket_helper(minio_server, datapath, "bodo-test", "us-east-1")


# A fixture such as the one below is run only on the first usage by a test
# So the bucket wouldn't be initialized until it is used by a test
# Similarly, once initialized, it remains in scope for the rest of the
# session and isn't re-initialized.
@pytest.fixture(scope="session")
def s3_bucket_us_west_2(minio_server, datapath):
    """
    creates a bucket called bodo-test-2 in s3 in the us-west-2 region and adds files to it
    this bucket will be useful in testing auto s3 region
    detection functionality in bodo
    """
    return s3_bucket_helper(minio_server, datapath, "bodo-test-2", "us-west-2")


@pytest.mark.hdfs
@pytest.fixture(scope="session")
def hadoop_server():
    """
    host and port of hadoop server
    """

    host = "localhost"
    port = 9000

    return host, port


@pytest.mark.hdfs
@pytest.fixture(scope="session")
def hdfs_dir(hadoop_server, datapath):
    """
    create a directory in hdfs and add files to it
    """
    hdfs3 = pytest.importorskip("hdfs3")
    from hdfs3 import HDFileSystem

    host, port = hadoop_server
    dir_name = "bodo-test"

    if bodo.get_rank() == 0:
        hdfs = HDFileSystem(host=host, port=port)
        hdfs.mkdir("/" + dir_name)
        test_hdfs_files = [
            ("csv_data1.csv", datapath("csv_data1.csv")),
            ("csv_data_date1.csv", datapath("csv_data_date1.csv")),
            ("asof1.pq", datapath("asof1.pq")),
            ("groupby3.pq", datapath("groupby3.pq")),
            ("example.json", datapath("example.json")),
        ]
        for fname, path in test_hdfs_files:
            formatted_fname = "/{}/{}".format(dir_name, fname)
            hdfs.put(path, formatted_fname)

        hdfs.mkdir("/bodo-test/int_nulls_multi.pq")
        prefix = datapath("int_nulls_multi.pq")
        pat = prefix + "/*.snappy.parquet"
        int_nulls_multi_parts = [f for f in glob.glob(pat)]
        for path in int_nulls_multi_parts:
            fname = path[len(prefix) + 1 :]
            fname = "/{}/int_nulls_multi.pq/{}".format(dir_name, fname)
            hdfs.put(path, fname)

        hdfs.mkdir("/bodo-test/example_single.json")
        prefix = datapath("example_single.json")
        pat = prefix + "/*.json"
        example_single_parts = [f for f in glob.glob(pat)]
        for path in example_single_parts:
            fname = path[len(prefix) + 1 :]
            fname = "/{}/example_single.json/{}".format(dir_name, fname)
            hdfs.put(path, fname)

        hdfs.mkdir("/bodo-test/example_multi.json")
        prefix = datapath("example_multi.json")
        pat = prefix + "/*.json"
        example_multi_parts = [f for f in glob.glob(pat)]
        for path in example_multi_parts:
            fname = path[len(prefix) + 1 :]
            fname = "/{}/example_multi.json/{}".format(dir_name, fname)
            hdfs.put(path, fname)

    bodo.barrier()
    return dir_name


@pytest.mark.hdfs
@pytest.fixture(scope="session")
def hdfs_datapath(hadoop_server, hdfs_dir):
    """
    Get the path to a test data file in hdfs
    """

    host, port = hadoop_server
    BASE_PATH = "hdfs://{}:{}/{}".format(host, port, hdfs_dir)

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        return path

    return deco


@pytest.fixture(scope="session", autouse=True)
def is_slow_run(request):
    """
    Return a flag on whether it is a slow test run (to skip some tests)
    """
    return "not slow" not in request.session.config.option.markexpr


def pytest_addoption(parser):
    """Used with caching tests, stores if the --is_cached flag was used when calling pytest
    into the pytestconfig"""

    # Minor check
    try:
        parser.addoption("--is_cached", action="store_true", default=False)
    except:
        pass


@pytest.fixture()
def is_cached(pytestconfig):
    """Fixture used with caching tests, returns true if pytest was called with --is_cached
    and false otherwise"""
    return pytestconfig.getoption("is_cached")


@pytest.fixture(scope="session")
@pytest.mark.iceberg
def iceberg_database():
    """
    Create and populate Iceberg test tables.
    """
    from bodo.tests.iceberg_database_helpers.create_all_tables import (
        create_all_tables,
    )

    comm = MPI.COMM_WORLD

    warehouse_loc = os.path.abspath(os.getcwd())

    database_schema_or_e = None
    if bodo.get_rank() == 0:
        try:
            database_schema_or_e = create_all_tables()
        except Exception as e:
            database_schema_or_e = e
    database_schema_or_e = comm.bcast(database_schema_or_e)
    if isinstance(database_schema_or_e, Exception):
        raise database_schema_or_e
    database_schema = database_schema_or_e

    yield database_schema, warehouse_loc

    if bodo.get_rank() == 0:
        import shutil

        dir_to_rm = os.path.join(warehouse_loc, database_schema)
        shutil.rmtree(dir_to_rm, ignore_errors=True)


@pytest.fixture(scope="session")
@pytest.mark.iceberg
def iceberg_table_conn():
    """Get the connection string and database-schema for Iceberg table.

    Parameters
    ----------
    table_name : str
    database_schema: str
    warehouse_loc: str

    Returns
    -------
    conn : connection string for the iceberg database

    Raises
    ------
    ValueError
        If the table doesn't exist.
    """

    def deco(
        table_name: str, database_schema: str, warehouse_loc: str, check_exists=True
    ):
        path = os.path.join(warehouse_loc, database_schema, table_name)
        if check_exists and not os.path.exists(path):
            msg = "Could not find table {}."
            raise ValueError(msg.format(table_name))
        # Currently the connection string is the location of the warehouse
        return f"iceberg://{warehouse_loc}"

    return deco


@pytest.fixture(
    params=(
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ),
    scope="session",
)
def cmp_op(request):
    return request.param
