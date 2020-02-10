# Copyright (C) 2019 Bodo Inc. All rights reserved.
import os
import gc
import pytest
import bodo
from numba.runtime import rtsys


# similar to Pandas
@pytest.fixture(scope="session")
def datapath():
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

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
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
    yield
    gc.collect()
    new = rtsys.get_allocation_stats()
    total_alloc = new.alloc - old.alloc
    total_free = new.free - old.free
    total_mi_alloc = new.mi_alloc - old.mi_alloc
    total_mi_free = new.mi_free - old.mi_free
    assert total_alloc == total_free
    assert total_mi_alloc == total_mi_free


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    Mark the first half of the tests with marker "firsthalf"
    """
    n = len(items)
    for item in items[0 : n // 2]:
        item.add_marker(pytest.mark.firsthalf)


@pytest.fixture(scope="session")
def minio_server():
    import subprocess

    host, port = "127.0.0.1", "9000"
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    address = "{}:{}".format(host, port)

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


@pytest.fixture(scope="session")
def s3_bucket(minio_server, datapath):
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
            region_name="us-east-1",
        )
        bucket = s3.Bucket("bodo-test")
        bucket.create()
        test_s3_files = [
            ("csv_data1.csv", datapath("csv_data1.csv")),
            ("csv_data_date1.csv", datapath("csv_data_date1.csv")),
            ("asof1.pq", datapath("asof1.pq")),
            ("groupby3.pq", datapath("groupby3.pq")),
        ]
        for s3_key, file_name in test_s3_files:
            s3.meta.client.upload_file(file_name, "bodo-test", s3_key)

    bodo.barrier()
    return "bodo-test"
