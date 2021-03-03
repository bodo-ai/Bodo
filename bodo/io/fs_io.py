# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
S3 & Hadoop file system supports, and file system dependent calls
"""
import glob
import os
import warnings
from urllib.parse import urlparse

import llvmlite.binding as ll
import numba
import numpy as np
from numba.core import types
from numba.extending import overload

import bodo
from bodo.io import csv_cpp
from bodo.libs.distributed_api import Reduce_Type
from bodo.libs.str_ext import unicode_to_utf8, unicode_to_utf8_and_len
from bodo.utils.typing import BodoError, BodoWarning

_csv_write = types.ExternalFunction(
    "csv_write",
    types.void(
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int64,
        types.bool_,
        types.voidptr,
    ),
)
ll.add_symbol("csv_write", csv_cpp.csv_write)


def get_s3_fs():
    """
    initialize S3FileSystem with credentials
    """
    try:
        import s3fs
    except:  # pragma: no cover
        raise BodoError("Reading from s3 requires s3fs currently.")

    custom_endpoint = os.environ.get("AWS_S3_ENDPOINT", None)
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

    # always use s3fs.S3FileSystem.clear_instance_cache()
    # before initializing S3FileSystem due to inconsistent file system
    # between to_parquet to read_parquet
    if custom_endpoint is not None and (
        aws_access_key_id is None or aws_secret_access_key is None
    ):  # pragma: no cover
        warnings.warn(
            BodoWarning(
                "Reading from s3 with custom_endpoint, "
                "but environment variables AWS_ACCESS_KEY_ID or "
                "AWS_SECRET_ACCESS_KEY is not set."
            )
        )
    s3fs.core.S3FileSystem.clear_instance_cache()
    fs = s3fs.core.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        client_kwargs={"endpoint_url": custom_endpoint},
    )

    return fs


# hdfs related functions(hdfs_list_dir_fnames) should be included in
# coverage once hdfs tests are included in CI
def get_hdfs_fs(path):  # pragma: no cover
    """
    initialize pyarrow.hdfs.HadoopFileSystem from path
    This function can be removed once arrow's new HadoopFileSystem is a subclass
    of pyarrow.filesystem.FileSystem, and use the hdfs returned from
    hdfs_list_dir_fnames.
    https://issues.apache.org/jira/browse/ARROW-7957
    """

    # this HadoopFileSystem is the deprecated file system of pyarrow
    # need this for pq.ParquetDataset
    # because the new HadoopFileSystem is not a subclass of
    # pyarrow.filesystem.FileSystem which causes an error
    from pyarrow.hdfs import HadoopFileSystem as HdFS

    options = urlparse(path)
    if options.scheme in ("abfs", "abfss"):
        # need to pass the full URI as host to libhdfs
        host = path
        if options.port is None:
            port = 0
        else:
            port = options.port
        user = None
    else:
        host = options.hostname
        port = options.port
        user = options.username
    # creates a new Hadoop file system from uri
    try:
        fs = HdFS(host=host, port=port, user=user)
    except Exception as e:
        raise BodoError("Hadoop file system cannot be created: {}".format(e))

    return fs


def s3_is_directory(fs, path):
    """
    Return whether s3 path is a directory or not
    """
    import botocore
    import s3fs

    try:
        path_info = fs.info(path)
        return path_info["Size"] == 0 and path_info["type"].lower() == "directory"
    except botocore.exceptions.NoCredentialsError as e:
        raise BodoError(
            f"S3 NoCredentialsError: {e}.\n"
            "Most likely cause: you haven't provided S3 credentials, "
            "neither through environment variables, nor through a local "
            "AWS setup that makes the credentials available at $HOME/.aws/credentials"
        )
    except s3fs.errors.ERROR_CODE_TO_EXCEPTION["AccessDenied"] as e:
        raise BodoError(
            f"S3 PermissionError: {e}. \n"
            "Most likely cause: your S3 credentials are incorrect or do not have the "
            "correct permissions."
        )
    except BodoError:  # pragma: no cover
        raise
    except Exception as e:  # pragma: no cover
        raise BodoError(f"error from s3fs: {type(e).__name__}: {str(e)}")


def s3_list_dir_fnames(fs, path):
    """
    If path is a directory, return all file names in the directory.
    This returns the base name without the path:
    ["file_name1", "file_name2", ...]
    If path is a file, return None
    """
    import botocore
    import s3fs

    file_names = None
    try:
        # check if path is a directory, and if there is a zero-size object
        # with the name of the directory. If there is, we have to omit it
        # because pq.ParquetDataset will throw Invalid Parquet file size is 0
        # bytes
        if s3_is_directory(fs, path):
            files = fs.ls(path)  # this is "s3://bucket/path-to-dir"
            if (
                files
                and (files[0] == path[5:] or files[0] == path[5:] + "/")
                and fs.info("s3://" + files[0])["Size"] == 0
            ):  # pragma: no cover
                # excluded from coverage because haven't found a reliable way
                # to create 0 size object that is a directory. For example:
                # fs.mkdir(path)  sometimes doesn't do anything at all
                # get actual names of objects inside the dir
                file_names = [fname[fname.rindex("/") + 1 :] for fname in files[1:]]
            else:
                file_names = [fname[fname.rindex("/") + 1 :] for fname in files]

    except botocore.exceptions.NoCredentialsError as e:
        raise BodoError(
            f"S3 NoCredentialsError: {e}.\n"
            "Most likely cause: you haven't provided S3 credentials, "
            "neither through environment variables, nor through a local "
            "AWS setup that makes the credentials available at $HOME/.aws/credentials"
        )
    except s3fs.errors.ERROR_CODE_TO_EXCEPTION["AccessDenied"] as e:
        raise BodoError(
            f"S3 PermissionError: {e}. \n"
            "Most likely cause: your S3 credentials are incorrect or do not have the "
            "correct permissions."
        )
    except BodoError:  # pragma: no cover
        raise
    except Exception as e:  # pragma: no cover
        raise BodoError(f"error from s3fs: {type(e).__name__}: {str(e)}")

    return file_names


def hdfs_is_directory(path):
    """
    Return whether hdfs path is a directory or not
    """
    # this HadoopFileSystem is the new file system of pyarrow
    from pyarrow.fs import FileType, HadoopFileSystem

    options = urlparse(path)
    hdfs_path = options.path  # path within hdfs(i.e. dir/file)

    try:
        hdfs = HadoopFileSystem.from_uri(path)
    except Exception as e:
        raise BodoError(" Hadoop file system cannot be created: {}".format(e))
    # target stat of the path: file or just the directory itself
    target_stat = hdfs.get_file_info([hdfs_path])

    if target_stat[0].type in (FileType.NotFound, FileType.Unknown):
        raise BodoError("{} is a " "non-existing or unreachable file".format(path))

    if (not target_stat[0].size) and target_stat[0].type == FileType.Directory:
        return hdfs, True

    return hdfs, False


def hdfs_list_dir_fnames(path):  # pragma: no cover
    """
    initialize pyarrow.fs.HadoopFileSystem from path
    If path is a directory, return all file names in the directory.
    This returns the base name without the path:
    ["file_name1", "file_name2", ...]
    If path is a file, return None
    return (pyarrow.fs.HadoopFileSystem, file_names)
    """

    from pyarrow.fs import FileSelector

    file_names = None
    hdfs, isdir = hdfs_is_directory(path)
    if isdir:
        options = urlparse(path)
        hdfs_path = options.path  # path within hdfs(i.e. dir/file)

        file_selector = FileSelector(hdfs_path, recursive=True)
        try:
            file_stats = hdfs.get_file_info(file_selector)
        except Exception as e:
            raise BodoError(
                "Exception on getting directory info " "of {}: {}".format(hdfs_path, e)
            )
        file_names = [file_stat.base_name for file_stat in file_stats]

    return (hdfs, file_names)


def abfs_is_directory(path):  # pragma: no cover
    """
    Return whether abfs path is a directory or not
    """

    hdfs = get_hdfs_fs(path)
    try:
        # target stat of the path: file or just the directory itself
        target_stat = hdfs.info(path)
    except OSError:
        raise BodoError("{} is a " "non-existing or unreachable file".format(path))

    if (target_stat["size"] == 0) and target_stat["kind"].lower() == "directory":
        return hdfs, True

    return hdfs, False


def abfs_list_dir_fnames(path):  # pragma: no cover
    """
    initialize pyarrow.fs.HadoopFileSystem from path
    If path is a directory, return all file names in the directory.
    This returns the base name without the path:
    ["file_name1", "file_name2", ...]
    If path is a file, return None
    return (pyarrow.fs.HadoopFileSystem, file_names)
    """

    file_names = None
    hdfs, isdir = abfs_is_directory(path)
    if isdir:
        options = urlparse(path)
        hdfs_path = options.path  # path within hdfs(i.e. dir/file)

        try:
            files = hdfs.ls(hdfs_path)
        except Exception as e:
            raise BodoError(
                "Exception on getting directory info " "of {}: {}".format(hdfs_path, e)
            )
        file_names = [fname[fname.rindex("/") + 1 :] for fname in files]

    return (hdfs, file_names)


def directory_of_files_common_filter(fname):
    # Ignore the same files as pyarrow,
    # https://github.com/apache/arrow/blob/4beb514d071c9beec69b8917b5265e77ade22fb3/python/pyarrow/parquet.py#L1039
    return not (
        fname.endswith(".crc")  # Checksums
        or fname.endswith("_$folder$")  # HDFS directories in S3
        or fname.startswith(".")  # Hidden files starting with .
        or fname.startswith("_")
        and fname != "_delta_log"  # Hidden files starting with _ skip deltalake
    )


def find_file_name_or_handler(path, ftype):
    """
    Find path_or_buf argument for pd.read_csv()/pd.read_json()

    If the path points to a single file:
        POSIX: file_name_or_handler = file name
        S3 & HDFS: file_name_or_handler = handler to the file
    If the path points to a directory:
        sort all non-empty files with the corresponding suffix
        POSIX: file_name_or_handler = file name of the first file in sorted files
        S3 & HDFS: file_name_or_handler = handler to the first file in sorted files

    Parameters:
        path: path to the object we are reading, this can be a file or a directory
        ftype: 'csv' or 'json'
    Returns:
        (is_handler, file_name_or_handler, f_size, fs)
        is_handler: True if file_name_or_handler is a handler,
                    False otherwise(file_name_or_handler is a file_name)
        file_name_or_handler: file_name or handler to pass to pd.read_csv()/pd.read_json()
        f_size: size of file_name_or_handler
        fs: file system for s3/hdfs
    """
    from urllib.parse import urlparse

    parsed_url = urlparse(path)
    fname = path
    fs = None
    func_name = "read_json" if ftype == "json" else "read_csv"
    err_msg = f"pd.{func_name}(): there is no {ftype} file in directory: {fname}"

    filter_func = directory_of_files_common_filter

    if parsed_url.scheme == "s3":
        is_handler = True
        fs = get_s3_fs()
        all_files = s3_list_dir_fnames(fs, path)  # can return None if not dir
        f_size = fs.info(fname)["size"]

        if all_files:
            path = path.rstrip("/")
            all_files = [
                (path + "/" + f) for f in sorted(filter(filter_func, all_files))
            ]
            all_csv_files = [f for f in all_files if fs.info(f)["size"] > 0]
            if len(all_csv_files) == 0:  # pragma: no cover
                # TODO: test
                raise BodoError(err_msg)
            fname = all_csv_files[0]
            f_size = fs.info(fname)["size"]
            fname = fname[5:]  # strip off s3://

        file_name_or_handler = fs.open(fname, "rb")
    elif parsed_url.scheme == "hdfs":  # pragma: no cover
        is_handler = True
        (fs, all_files) = hdfs_list_dir_fnames(path)
        f_size = fs.get_file_info([parsed_url.path])[0].size

        if all_files:
            path = path.rstrip("/")
            all_files = [
                (path + "/" + f) for f in sorted(filter(filter_func, all_files))
            ]
            all_csv_files = [
                f for f in all_files if fs.get_file_info([urlparse(f).path])[0].size > 0
            ]
            if len(all_csv_files) == 0:  # pragma: no cover
                # TODO: test
                raise BodoError(err_msg)
            fname = all_csv_files[0]
            fname = urlparse(fname).path  # strip off hdfs://port:host/
            f_size = fs.get_file_info([fname])[0].size

        file_name_or_handler = fs.open_input_file(fname)
    # TODO: this can be merged with hdfs path above when pyarrow's new
    # HadoopFileSystem wrapper supports abfs scheme
    elif parsed_url.scheme in ("abfs", "abfss"):  # pragma: no cover
        is_handler = True
        (fs, all_files) = abfs_list_dir_fnames(path)
        f_size = fs.info(fname)["size"]

        if all_files:
            path = path.rstrip("/")
            all_files = [
                (path + "/" + f) for f in sorted(filter(filter_func, all_files))
            ]
            all_csv_files = [f for f in all_files if fs.info(f)["size"] > 0]
            if len(all_csv_files) == 0:  # pragma: no cover
                # TODO: test
                raise BodoError(err_msg)
            fname = all_csv_files[0]
            f_size = fs.info(fname)["size"]
            fname = urlparse(fname).path  # strip off abfs[s]://port:host/

        file_name_or_handler = fs.open(fname, "rb")
    else:
        assert parsed_url.scheme == ""
        is_handler = False

        if os.path.isdir(path):
            files = filter(filter_func, glob.glob(os.path.join(path, "*")))
            all_csv_files = [f for f in sorted(files) if os.path.getsize(f) > 0]
            if len(all_csv_files) == 0:  # pragma: no cover
                # TODO: test
                raise BodoError(err_msg)
            fname = all_csv_files[0]

        f_size = os.path.getsize(fname)
        file_name_or_handler = fname

    # although fs is never used, we need to return it so that s3/hdfs
    # connections are not closed
    return is_handler, file_name_or_handler, f_size, fs


def get_s3_bucket_region(s3_filepath):
    """
    Get the region of the s3 bucket from a s3 url of type s3://<BUCKET_NAME>/<FILEPATH>.
    PyArrow's region detection only works for actual S3 buckets.
    Returns an empty string in case region cannot be determined.
    """

    try:
        from pyarrow import fs as pa_fs
    except:  # pragma: no cover
        raise BodoError("Reading from s3 requires pyarrow currently.")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    bucket_loc = None
    if bodo.get_rank() == 0:
        try:
            temp_fs, _ = pa_fs.S3FileSystem.from_uri(s3_filepath)
            bucket_loc = temp_fs.region
        except Exception as e:  # pragma: no cover
            if os.environ.get("AWS_DEFAULT_REGION", "") == "":
                warnings.warn(
                    BodoWarning(
                        f"Unable to get S3 Bucket Region.\n{e}.\nValue not defined in the AWS_DEFAULT_REGION environment variable either. Region defaults to us-east-1 currently."
                    )
                )
            bucket_loc = ""
    bucket_loc = comm.bcast(bucket_loc)

    return bucket_loc


@numba.njit()
def get_s3_bucket_region_njit(s3_filepath):  # pragma: no cover
    """
    njit wrapper around get_s3_bucket_region
    """
    with numba.objmode(bucket_loc="unicode_type"):
        bucket_loc = ""
        if s3_filepath.startswith("s3://"):
            bucket_loc = get_s3_bucket_region(s3_filepath)
    return bucket_loc


def csv_write(path_or_buf, D, is_parallel=False):  # pragma: no cover
    # This is a dummy function used to allow overload.
    return None


@overload(csv_write, no_unliteral=True)
def csv_write_overload(path_or_buf, D, is_parallel=False):
    def impl(path_or_buf, D, is_parallel=False):  # pragma: no cover
        # Assuming that path_or_buf is a string
        bucket_region = get_s3_bucket_region_njit(path_or_buf)
        # TODO: support non-ASCII file names?
        utf8_str, utf8_len = unicode_to_utf8_and_len(D)
        offset = 0
        if is_parallel:
            offset = bodo.libs.distributed_api.dist_exscan(
                utf8_len, np.int32(Reduce_Type.Sum.value)
            )
        _csv_write(
            unicode_to_utf8(path_or_buf),
            utf8_str,
            offset,
            utf8_len,
            is_parallel,
            unicode_to_utf8(bucket_region),
        )
        # Check if there was an error in the C++ code. If so, raise it.
        bodo.utils.utils.check_and_propagate_cpp_exception()

    return impl
