from __future__ import annotations

import datetime
import gc
import glob
import hashlib
import operator
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Generator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
)
from uuid import uuid4

import adlfs
import pandas as pd
import psutil
import pytest
from mpi4py import MPI
from numba.core.runtime import rtsys  # noqa TID253

import bodo
import bodo.user_logging
from bodo.tests.iceberg_database_helpers.utils import DATABASE_NAME
from bodo.tests.utils import temp_env_override

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


# Disable broadcast join as the default
os.environ["BODO_BCAST_JOIN_THRESHOLD"] = "0"


# Similar to Pandas
class DataPath(Protocol):
    def __call__(self, *args: str, check_exists: bool = True) -> str: ...


def datapath_util(*args, base_path=None, check_exists=True) -> str:
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(base_path, *args)
    if check_exists and not os.path.exists(path):
        msg = "Could not find file {}."
        raise ValueError(msg.format(path))
    return path


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
        return datapath_util(*args, base_path=BASE_PATH, check_exists=check_exists)

    return deco


@pytest.fixture(scope="session", autouse=True)
def enable_numba_alloc_stats():
    """Enable Numba's allocation stat collection for memory_leak_check below"""
    from numba.core.runtime import _nrt_python

    _nrt_python.memsys_enable_stats()


@pytest.fixture(scope="function")
def memory_leak_check():
    """
    A context manager fixture that makes sure there is no memory leak in the test.
    Equivalent to Numba's MemoryLeakMixin:
    https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/tests/support.py#L688
    """
    import bodo.tests.utils
    import bodo.utils.allocation_tracking

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

    # Don't check for memory leaks if the test is being re-run by flaky
    if bodo.tests.utils.pytest_snowflake_is_rerunning:
        bodo.tests.utils.pytest_snowflake_is_rerunning = False
    else:
        assert total_alloc == total_free
        assert total_mi_alloc == total_mi_free


@pytest.fixture(scope="function")
def jit_import_check():
    """Fixture used to assert that a test should not import JIT."""
    jit_already_imported = "bodo.decorators" in sys.modules
    yield
    assert jit_already_imported or "bodo.decorators" not in sys.modules, (
        "Test was not explicitly marked as a JIT dependency, but imported JIT."
    )


def item_file_name(item):
    """Get the name of a pytest item. Uses the default pytest implementation, except for C++ tests, where we return the cached name"""
    if isinstance(item, (CppTestFile, CppTestItem)):
        return item.filename
    else:
        return item.module.__name__.split(".")[-1] + ".py"


def item_module_name(item):
    """Get the pytest module name. Uses default pytest implementation, except for c++, whiche uses the filename"""
    if isinstance(item, (CppTestFile, CppTestItem)):
        return item.filename
    else:
        return item.module.__name__


def unique_test_name(test):
    """Return a unique string for every pytest function"""
    return item_module_name(test) + str(test)


def get_last_byte_of_test_hash(test):
    """
    Gets the last byte of hashing the test name. The hash function doesn't
    matter as long as it distributes tests reasonably well across all the
    buckets.
    """
    name = unique_test_name(test).encode("utf-8")
    return hashlib.sha1(name).digest()[-1]


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    """
    azure_1p_markers = [
        pytest.mark.bodo_1of22,
        pytest.mark.bodo_2of22,
        pytest.mark.bodo_3of22,
        pytest.mark.bodo_4of22,
        pytest.mark.bodo_5of22,
        pytest.mark.bodo_6of22,
        pytest.mark.bodo_7of22,
        pytest.mark.bodo_8of22,
        pytest.mark.bodo_9of22,
        pytest.mark.bodo_10of22,
        pytest.mark.bodo_11of22,
        pytest.mark.bodo_12of22,
        pytest.mark.bodo_13of22,
        pytest.mark.bodo_14of22,
        pytest.mark.bodo_15of22,
        pytest.mark.bodo_16of22,
        pytest.mark.bodo_17of22,
        pytest.mark.bodo_18of22,
        pytest.mark.bodo_19of22,
        pytest.mark.bodo_20of22,
        pytest.mark.bodo_21of22,
        pytest.mark.bodo_22of22,
    ]
    azure_2p_markers = [
        pytest.mark.bodo_1of30,
        pytest.mark.bodo_2of30,
        pytest.mark.bodo_3of30,
        pytest.mark.bodo_4of30,
        pytest.mark.bodo_5of30,
        pytest.mark.bodo_6of30,
        pytest.mark.bodo_7of30,
        pytest.mark.bodo_8of30,
        pytest.mark.bodo_9of30,
        pytest.mark.bodo_10of30,
        pytest.mark.bodo_11of30,
        pytest.mark.bodo_12of30,
        pytest.mark.bodo_13of30,
        pytest.mark.bodo_14of30,
        pytest.mark.bodo_15of30,
        pytest.mark.bodo_16of30,
        pytest.mark.bodo_17of30,
        pytest.mark.bodo_18of30,
        pytest.mark.bodo_19of30,
        pytest.mark.bodo_20of30,
        pytest.mark.bodo_21of30,
        pytest.mark.bodo_22of30,
        pytest.mark.bodo_23of30,
        pytest.mark.bodo_24of30,
        pytest.mark.bodo_25of30,
        pytest.mark.bodo_26of30,
        pytest.mark.bodo_27of30,
        pytest.mark.bodo_28of30,
        pytest.mark.bodo_29of30,
        pytest.mark.bodo_30of30,
    ]

    # DataFrame library NP=1
    azure_df_1p_markers = [pytest.mark.bodo_df_1of2, pytest.mark.bodo_df_2of2]

    # DataFrame library NP=2
    azure_df_2p_markers = [
        pytest.mark.bodo_df_1of5,
        pytest.mark.bodo_df_2of5,
        pytest.mark.bodo_df_3of5,
        pytest.mark.bodo_df_4of5,
        pytest.mark.bodo_df_5of5,
    ]

    # Spawn NP=2
    azure_spawn_2p_markers = [pytest.mark.bodo_spawn_1of2, pytest.mark.bodo_spawn_2of2]

    test_splits = [
        azure_1p_markers,
        azure_2p_markers,
        azure_df_1p_markers,
        azure_df_2p_markers,
        azure_spawn_2p_markers,
    ]

    # BODO_TEST_PYTEST_MOD environment variable indicates that we only want
    # to run the tests from the given test file. In this case, we add the
    # "single_mod" mark to the tests belonging to that module. This envvar is
    # set in runtests.py, which also adds the "-m single_mod" to the pytest
    # command (thus ensuring that only those tests run)
    module_to_run = os.environ.get("BODO_TEST_PYTEST_MOD", None)
    if module_to_run is not None:
        for item in items:
            if module_to_run == item_file_name(item):
                item.add_marker(pytest.mark.single_mod)

    # If this assert fails, we need to get more than just the last byte from the
    # hash below, and then the limit can be updated to 2**(8 * num_bytes).
    assert all(len(split) < 128 for split in test_splits), (
        "Need more bytes from hash to distribute tests"
    )

    for item in items:
        hash_ = get_last_byte_of_test_hash(item)
        # Divide the tests evenly so long tests don't end up in 1 group
        markers = [split[hash_ % len(split)] for split in test_splits]
        # All of the test_s3.py tests must be on the same rank because they
        # haven't been refactored to remove cross-test dependencies.
        testfile = item_file_name(item)
        if "test_s3.py" in testfile:
            markers = [split[0] for split in test_splits]

        for marker in markers:
            item.add_marker(marker)

    # If running tests in DataFrames mode, add a check that JIT is not imported to all tests
    # that don't explicitly depend on JIT via the jit_dependency marker.
    if bodo.test_dataframe_library_enabled:
        assert "bodo.decorators" not in sys.modules, (
            "JIT was imported before pytest_collection_modifyitems finished."
        )

        no_jit_dep = [
            item for item in items if not item.get_closest_marker("jit_dependency")
        ]

        for item in no_jit_dep:
            if (
                hasattr(item, "fixturenames")
                and "jit_import_check" not in item.fixturenames
            ):
                item.fixturenames = ["jit_import_check"] + item.fixturenames


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
    hash_val = hashlib.sha1(testname.encode("utf-8")).digest()
    int_hash = int.from_bytes(hash_val) % num_groups
    return str(int_hash)


@pytest.fixture(scope="session")
def minio_server():
    """
    Spins up minio server
    """
    cwd = os.getcwd()

    # Kill an existing Minio server (orphan process from segfault)
    for proc in psutil.process_iter():
        if proc.name() == "minio":
            proc.kill()
            shutil.rmtree(cwd + "/Data", ignore_errors=True)
            time.sleep(1)

    # Session level environment variables used for S3 Testing.

    host, port = "127.0.0.1", "9000"
    access_key = "bodotest1"
    secret_key = "bodosecret1"
    address = f"{host}:{port}"

    os.environ["MINIO_ROOT_USER"] = access_key
    os.environ["MINIO_ROOT_PASSWORD"] = secret_key
    # For compatibility with older MinIO versions.
    os.environ["MINIO_ACCESS_KEY"] = access_key
    os.environ["MINIO_SECRET_KEY"] = secret_key

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
    except OSError:
        pytest.skip("`minio` command cannot be located")
    else:
        yield access_key, secret_key, address
    finally:
        if bodo.get_rank() == 0:
            if proc is not None:
                proc.kill()
            shutil.rmtree(cwd + "/Data", ignore_errors=True)


@pytest.fixture(scope="function")
def minio_server_with_s3_envs(minio_server: tuple[str, str, str]):
    with temp_env_override(
        {
            "AWS_ACCESS_KEY_ID": minio_server[0],
            "AWS_SECRET_ACCESS_KEY": minio_server[1],
            "AWS_SESSION_TOKEN": None,
            "AWS_S3_ENDPOINT": f"http://{minio_server[2]}/",
        }
    ):
        yield minio_server


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
            endpoint_url=f"http://{address}/",
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
                bucket.upload_file(path, fname)

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
                # Avoid "\" generated on Windows that causes object name errors
                s3.meta.client.upload_file(
                    full_path, bucket_name, rel_path.replace("\\", "/")
                )

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


@pytest.fixture(scope="session")
def hadoop_server():
    """
    host and port of hadoop server
    """

    host = "localhost"
    port = 9000

    return host, port


@pytest.fixture(scope="session")
def hdfs_dir(hadoop_server, datapath):
    """
    create a directory in hdfs and add files to it
    """
    from pyarrow.fs import HadoopFileSystem as HDFileSystem
    from pyarrow.fs import LocalFileSystem, copy_files

    host, port = hadoop_server
    dir_name = "bodo-test"

    if bodo.get_rank() == 0:
        hdfs = HDFileSystem(host=host, port=port)
        local_fs = LocalFileSystem()
        hdfs.create_dir("/" + dir_name)
        test_hdfs_files = [
            ("csv_data1.csv", datapath("csv_data1.csv")),
            ("csv_data_date1.csv", datapath("csv_data_date1.csv")),
            ("asof1.pq", datapath("asof1.pq")),
            ("groupby3.pq", datapath("groupby3.pq")),
            ("example.json", datapath("example.json")),
        ]
        for fname, path in test_hdfs_files:
            formatted_fname = f"/{dir_name}/{fname}"
            copy_files(path, formatted_fname, local_fs, hdfs)

        hdfs.create_dir("/bodo-test/int_nulls_multi.pq")
        prefix = datapath("int_nulls_multi.pq")
        pat = prefix + "/*.snappy.parquet"
        int_nulls_multi_parts = list(glob.glob(pat))
        for path in int_nulls_multi_parts:
            fname = path[len(prefix) + 1 :]
            fname = f"/{dir_name}/int_nulls_multi.pq/{fname}"
            copy_files(path, fname, local_fs, hdfs)

        hdfs.create_dir("/bodo-test/example_single.json")
        prefix = datapath("example_single.json")
        pat = prefix + "/*.json"
        example_single_parts = list(glob.glob(pat))
        for path in example_single_parts:
            fname = path[len(prefix) + 1 :]
            fname = f"/{dir_name}/example_single.json/{fname}"
            copy_files(path, fname, local_fs, hdfs)

        hdfs.create_dir("/bodo-test/example_multi.json")
        prefix = datapath("example_multi.json")
        pat = prefix + "/*.json"
        example_multi_parts = list(glob.glob(pat))
        for path in example_multi_parts:
            fname = path[len(prefix) + 1 :]
            fname = f"/{dir_name}/example_multi.json/{fname}"
            copy_files(path, fname, local_fs, hdfs)

    bodo.barrier()
    return dir_name


@pytest.fixture(scope="session")
def hdfs_datapath(hadoop_server, hdfs_dir):
    """
    Get the path to a test data file in hdfs
    """

    host, port = hadoop_server
    BASE_PATH = f"hdfs://{host}:{port}/{hdfs_dir}"

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
    except Exception:
        pass


@pytest.fixture()
def is_cached(pytestconfig):
    """Fixture used with caching tests, returns true if pytest was called with --is_cached
    and false otherwise"""
    return pytestconfig.getoption("is_cached")


@pytest.fixture(scope="session")
def iceberg_database() -> Generator[
    Callable[[list[str] | str], tuple[str, str]], None, None
]:
    """
    Create and populate Iceberg test tables.
    """
    from bodo.tests.iceberg_database_helpers.create_tables import (
        create_tables,
    )

    comm = MPI.COMM_WORLD

    warehouse_loc = os.path.abspath(os.getcwd())

    # Use a list to store the database schema, so that it can be accessed by the fixture
    database_schema = []

    # TODO(aneesh) this should probably take in a spark instance explicitly to
    # make this safer to use - calling this function will invalidate all old
    # spark referneces.
    def create_tables_on_rank_one(
        tables: list[str] | str = [], spark: SparkSession | None = None
    ) -> tuple[str, str]:
        if not isinstance(tables, list):
            tables = [tables]
        database_schema_or_e = None
        if bodo.get_rank() == 0:
            try:
                database_schema_or_e = create_tables(tables, spark=spark)
            except Exception as e:
                database_schema_or_e = e
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
        database_schema_or_e = comm.bcast(database_schema_or_e)
        if isinstance(database_schema_or_e, Exception):
            raise database_schema_or_e
        if database_schema_or_e not in database_schema:
            database_schema.append(database_schema_or_e)
        return database_schema_or_e, warehouse_loc

    yield create_tables_on_rank_one

    bodo.barrier()
    if bodo.get_rank() == 0:
        import shutil

        dir_to_rm = os.path.join(warehouse_loc, DATABASE_NAME)
        shutil.rmtree(dir_to_rm, ignore_errors=True)


@pytest.fixture(scope="session")
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


@pytest.fixture
def time_df():
    """
    Fixture containing a representative set of bodo.types.Time object
    for use in testing, including None object.
    """
    return {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        bodo.types.Time(17, 33, 26, 91, 8),
                        bodo.types.Time(0, 24, 43, 365, 18),
                        bodo.types.Time(3, 59, 6, 25, 757),
                        bodo.types.Time(),
                        bodo.types.Time(4),
                        bodo.types.Time(6, 41),
                        bodo.types.Time(22, 13, 57),
                        bodo.types.Time(17, 34, 29, 90),
                        bodo.types.Time(7, 3, 45, 876, 234),
                        None,
                    ]
                ),
                "B": pd.Series(
                    [
                        bodo.types.Time(20, 6, 26, 324, 4),
                        bodo.types.Time(3, 59, 6, 25, 57),
                        bodo.types.Time(7, 3, 45, 876, 234),
                        bodo.types.Time(17, 34, 29, 90),
                        bodo.types.Time(22, 13, 57),
                        bodo.types.Time(6, 41),
                        bodo.types.Time(4),
                        bodo.types.Time(),
                        None,
                        bodo.types.Time(0, 24, 4, 512, 18),
                    ]
                ),
            }
        )
    }


@pytest.fixture
def date_df():
    """
    Fixture containing a representative set of datetime.date object
    for use in testing, including None object.
    """
    return {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        datetime.date(2017, 3, 26),
                        datetime.date(2000, 12, 31),
                        datetime.date(2003, 9, 6),
                        datetime.date(2023, 3, 6),
                        datetime.date(2024, 1, 1),
                        datetime.date(1996, 4, 25),
                        datetime.date(2022, 11, 17),
                        datetime.date(1917, 7, 29),
                        datetime.date(2007, 10, 14),
                        None,
                    ]
                ),
                "B": pd.Series(
                    [
                        datetime.date(2020, 6, 26),
                        datetime.date(2025, 5, 3),
                        datetime.date(1987, 3, 15),
                        datetime.date(2117, 8, 29),
                        datetime.date(1822, 12, 7),
                        datetime.date(1906, 4, 14),
                        datetime.date(2004, 9, 13),
                        datetime.date(1917, 7, 29),
                        None,
                        datetime.date(1700, 2, 4),
                    ]
                ),
                "C": pd.Series(
                    [
                        datetime.date(2012, 4, 22),
                        datetime.date(2043, 11, 3),
                        datetime.date(1998, 9, 11),
                        datetime.date(2100, 4, 19),
                        datetime.date(1832, 1, 7),
                    ]
                    * 2
                ),
            }
        )
    }


class CppTestFile(pytest.File):
    """Represents a c++ unit testing file that integrates with pytest.

    When bodo is built with `setup.py develop --test`, the bodo.ext module contains the
    compiled versions of the C++ tests. When pytest is run with a .cpp file, this class makes it
    look up the c++ file in the compiled module. The module is able to enumerate all tests in the file.
    """

    @classmethod
    def from_parent(cls, parent, **kwargs):
        return super().from_parent(parent, **kwargs)

    def __init__(self, parent, **kwargs):
        super().__init__(path=kwargs["path"], parent=parent)
        self.tests = kwargs["tests"]
        self.filename = kwargs["filename"]

    def collect(self):
        for test in self.tests:
            yield CppTestItem.from_parent(self, test=test)


class CppTestItem(pytest.Item):
    """Every individual c++ test in a file can be enumerated. The 'CppTestFile' class builds
    a 'CppTestItem' for each test in the given c++ file.
    """

    def __init__(self, parent, *, test):
        super().__init__(test.name, parent=parent)
        self.test = test
        for marker in self.test.markers:
            self.add_marker(marker)

    @property
    def filename(self):
        return self.parent.filename

    def runtest(self):
        self.test()

    def reportinfo(self):
        return self.test.filename, self.test.lineno, self.test.name


def pytest_collect_file(parent, file_path: Path):
    """
    A hook into py.test to collect test_*.cpp test files.

    This adds functionality to collect test_*.cpp files (except for test_framework.cpp).

    Each c++ file gets a corresponding 'CppTestFile' pytest item, which lets pytest run c++ tests.

    The C++ tests are built as part of the bodo.ext module when `--test` is given to the `setup.py develop` command.

    When built, the tests are available as the `bodo.ext.test_cpp` module. Otherwise, the import fails
    and this file won't collect any C++ tests (but it will print out a warning)
    """
    try:
        from bodo.ext import test_cpp
    except ImportError:
        test_cpp = None

    if (
        file_path.suffix == ".cpp"
        and file_path.name.startswith("test_")
        and file_path.name != "test_framework.cpp"
    ):
        if test_cpp is None:
            import warnings

            warnings.warn(
                "c++ tests are disabled in this build. Run 'setup.py develop' with the '--test' flag to enable."
            )
            return

        tests = [
            test
            for test in test_cpp.tests
            if file_path.name == os.path.basename(test.filename)
        ]
        return CppTestFile.from_parent(
            parent, path=file_path, filename=file_path.name, tests=tests
        )


@pytest.fixture(
    params=[
        "quarter",
        "yyy",
        pytest.param("MONTH", marks=pytest.mark.slow),
        "mon",
        "WEEK",
        pytest.param("wk", marks=pytest.mark.slow),
        pytest.param("DAY", marks=pytest.mark.slow),
        "dd",
    ]
)
def day_part_strings(request):
    """
    Fixture containing a representative set of large time unit part strings
    (larger or equal to day) for use in testing, including aliases.
    """
    return request.param


@pytest.fixture(
    params=[
        "HOUR",
        pytest.param("hr", marks=pytest.mark.slow),
        pytest.param("MINUTE", marks=pytest.mark.slow),
        "min",
        "SECOND",
        "ms",
        pytest.param("microsecond", marks=pytest.mark.slow),
        pytest.param("usec", marks=pytest.mark.slow),
        "nanosecs",
    ]
)
def time_part_strings(request):
    """
    Fixture containing a representative set of small time unit part strings
    (smaller or equal to hour) for use in testing, including aliases.
    """
    return request.param


@pytest.fixture(
    params=[
        "quarter",
        pytest.param("yyyy", marks=pytest.mark.slow),
        "week",
        pytest.param("mm", marks=pytest.mark.slow),
        "days",
        pytest.param("hour", marks=pytest.mark.slow),
        "minute",
        pytest.param("S", marks=pytest.mark.slow),
        "ms",
        pytest.param("us", marks=pytest.mark.slow),
        "nsecond",
    ]
)
def datetime_part_strings(request):
    """
    Fixture containing a representative set of datetime part strings
    for use in testing, including aliases.
    """
    return request.param


@pytest.fixture(scope="session")
def abfs_fs():
    """
    Create an Azure Blob FileSystem instance for testing.
    """

    account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    return adlfs.AzureBlobFileSystem(account_name=account_name, account_key=account_key)


@pytest.fixture
def tmp_abfs_path(abfs_fs):
    """
    Create a temporary ABFS path for testing.
    """
    from bodo.spawn.utils import run_rank0

    @run_rank0
    def setup():
        folder_name = str(uuid4())
        abfs_fs.mkdir(f"engine-unit-tests-tmp-blob/{folder_name}")
        return folder_name

    # Need to include account name in path for C++ filesystem code
    folder_name = setup()
    account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    yield f"abfs://engine-unit-tests-tmp-blob@{account_name}.dfs.core.windows.net/{folder_name}/"

    @run_rank0
    def cleanup():
        if abfs_fs.exists(f"engine-unit-tests-tmp-blob/{folder_name}"):
            abfs_fs.rm(f"engine-unit-tests-tmp-blob/{folder_name}", recursive=True)

    cleanup()


@pytest.fixture(scope="module")
def verbose_mode_on():
    bodo.set_verbose_level(2)
    yield
    bodo.user_logging.restore_default_bodo_verbose_level()
