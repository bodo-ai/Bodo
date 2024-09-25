# Copyright (C) 2022 Bodo Inc. All rights reserved.
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
from fsspec.implementations.arrow import (
    ArrowFile,
    ArrowFSWrapper,
    wrap_exceptions,
)
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    models,
    overload,
    register_model,
    unbox,
)

import bodo
from bodo.ext import arrow_cpp
from bodo.io import csv_cpp
from bodo.io.pyfs import get_pyarrow_fs_from_ptr
from bodo.libs.distributed_api import Reduce_Type
from bodo.libs.str_ext import unicode_to_utf8, unicode_to_utf8_and_len
from bodo.utils.typing import BodoError, BodoWarning, get_overload_constant_dict
from bodo.utils.utils import AWSCredentials, check_java_installation


# ----- monkey-patch fsspec.implementations.arrow.ArrowFSWrapper._open --------
def fsspec_arrowfswrapper__open(self, path, mode="rb", block_size=None, **kwargs):
    if mode == "rb":
        try:  # Bodo change: try to open the file for random access first
            # We need random access to read parquet file metadata
            stream = self.fs.open_input_file(path)
        except:  # pragma: no cover
            stream = self.fs.open_input_stream(path)
    elif mode == "wb":  # pragma: no cover
        stream = self.fs.open_output_stream(path)
    else:
        raise ValueError(f"unsupported mode for Arrow filesystem: {mode!r}")

    return ArrowFile(self, stream, path, mode, block_size, **kwargs)


ArrowFSWrapper._open = wrap_exceptions(fsspec_arrowfswrapper__open)
# -----------------------------------------------------------------------------


_csv_write = types.ExternalFunction(
    "csv_write",
    types.void(
        types.voidptr,  # char *_path_name
        types.voidptr,  # char *buff
        types.int64,  # int64_t start
        types.int64,  # int64_t count
        types.bool_,  # bool is_parallel
        types.voidptr,  # char *bucket_region
        types.voidptr,  # char *prefix
    ),
)
ll.add_symbol("csv_write", csv_cpp.csv_write)

bodo_error_msg = """
    Some possible causes:
        (1) Incorrect path: Specified file/directory doesn't exist or is unreachable.
        (2) Missing credentials: You haven't provided S3 credentials, neither through
            environment variables, nor through a local AWS setup
            that makes the credentials available at ~/.aws/credentials.
        (3) Incorrect credentials: Your S3 credentials are incorrect or do not have
            the correct permissions.
        (4) Wrong bucket region is used. Set AWS_DEFAULT_REGION variable with correct bucket region.
    """


def get_proxy_uri_from_env_vars():
    """
    Get proxy URI from environment variables if they're set,
    else return None.
    Precedence order of the different environment
    variables should be consistent with
    get_s3_proxy_options_from_env_vars in _s3_reader.cpp
    to avoid differences in compile-time and runtime
    behavior.
    """
    return (
        os.environ.get("http_proxy", None)
        or os.environ.get("https_proxy", None)
        or os.environ.get("HTTP_PROXY", None)
        or os.environ.get("HTTPS_PROXY", None)
    )


def validate_s3fs_installed():
    """
    Validate that s3fs is installed. An error is raised
    when this is not the case.
    """
    try:
        import s3fs  # noqa
    except ImportError:
        raise BodoError(
            "Couldn't import s3fs, which is required for certain types of S3 access."
            " s3fs can be installed by calling"
            " 'conda install -c conda-forge s3fs'.\n"
        )


def validate_gcsfs_installed():
    """
    Validate that gcsfs is installed. An error is raised
    when this is not the case.
    """
    try:
        import gcsfs  # noqa
    except ImportError:
        raise BodoError(
            "Couldn't import gcsfs, which is required for Google cloud access."
            " gcsfs can be installed by calling"
            " 'conda install -c conda-forge gcsfs'.\n"
        )


def get_s3_fs(
    region=None, storage_options=None, aws_credentials: AWSCredentials | None = None
):
    """
    initialize S3FileSystem with credentials
    """
    from pyarrow.fs import S3FileSystem

    custom_endpoint = os.environ.get("AWS_S3_ENDPOINT", None)
    if not region:
        region = os.environ.get("AWS_DEFAULT_REGION", None)

    anon = False
    proxy_options = get_proxy_uri_from_env_vars()
    if storage_options:
        anon = storage_options.get("anon", False)

    return S3FileSystem(
        anonymous=anon,
        region=region,
        endpoint_override=custom_endpoint,
        proxy_options=proxy_options,
        access_key=aws_credentials.access_key if aws_credentials else None,
        secret_key=aws_credentials.secret_key if aws_credentials else None,
        session_token=aws_credentials.session_token if aws_credentials else None,
    )


def get_s3_subtree_fs(bucket_name, region=None, storage_options=None):
    """
    Initialize S3 SubTreeFileSystem with credentials.
    When reading metadata or data from a dataset consisting of multiple
    files, we need to use a SubTreeFileSystem so that Arrow can speed
    up IO using multiple threads (file read ahead).
    In normal circumstances Arrow would create this automatically
    from a S3 URL, but to pass custom endpoint and use anonymous
    option we need to do this manually.
    """
    from pyarrow._fs import SubTreeFileSystem
    from pyarrow._s3fs import S3FileSystem

    custom_endpoint = os.environ.get("AWS_S3_ENDPOINT", None)
    if not region:
        region = os.environ.get("AWS_DEFAULT_REGION", None)

    anon = False
    proxy_options = get_proxy_uri_from_env_vars()
    if storage_options:
        anon = storage_options.get("anon", False)

    fs = S3FileSystem(
        region=region,
        endpoint_override=custom_endpoint,
        anonymous=anon,
        proxy_options=proxy_options,
    )
    return SubTreeFileSystem(bucket_name, fs)


def get_s3_fs_from_path(
    path,
    parallel=False,
    storage_options=None,
):
    """
    Get a pyarrow.fs.S3FileSystem object from an S3
    path, i.e. determine the region and
    create a FS for that region.
    The parallel option is passed on to the region detection code.
    This function is usually called on just rank 0 during compilation,
    hence parallel=False by default.
    """
    region = get_s3_bucket_region_njit(path, parallel=parallel)
    if region == "":
        region = None
    return get_s3_fs(region, storage_options)


# hdfs related functions(hdfs_list_dir_fnames) should be included in
# coverage once hdfs tests are included in CI
def get_hdfs_fs(path):  # pragma: no cover
    """
    initialize pyarrow.fs.HadoopFileSystem from path
    """

    from pyarrow.fs import HadoopFileSystem as HdFS

    options = urlparse(path)
    if options.scheme in ("abfs", "abfss"):
        # need to pass the full URI as host to libhdfs
        host = path
        user = None
    else:
        host = options.hostname
        user = options.username
    if options.port is None:
        port = 0
    else:
        port = options.port
    # creates a new Hadoop file system from uri
    try:
        fs = HdFS(host=host, port=port, user=user)
    except Exception as e:
        raise BodoError("Hadoop file system cannot be created: {}".format(e))

    return fs


def gcs_is_directory(path):
    import gcsfs

    fs = gcsfs.GCSFileSystem(token=None)
    try:
        isdir = fs.isdir(path)
    except gcsfs.utils.HttpError as e:
        raise BodoError(f"{e}. Make sure your google cloud credentials are set!")
    return isdir


def gcs_list_dir_fnames(path):
    import gcsfs

    fs = gcsfs.GCSFileSystem(token=None)
    return [f.split("/")[-1] for f in fs.ls(path)]


def s3_is_directory(fs, path):
    """
    Return whether s3 path is a directory or not
    """
    from pyarrow import fs as pa_fs

    try:
        options = urlparse(path)
        # Remove the s3:// prefix if it exists (and other path sanitization)
        path_ = (options.netloc + options.path).rstrip("/")
        path_info = fs.get_file_info(path_)
        if path_info.type in (pa_fs.FileType.NotFound, pa_fs.FileType.Unknown):
            raise FileNotFoundError(
                "{} is a non-existing or unreachable file".format(path)
            )
        if (not path_info.size) and path_info.type == pa_fs.FileType.Directory:
            return True
        return False
    except (FileNotFoundError, OSError) as e:
        raise
    except BodoError:  # pragma: no cover
        raise
    except Exception as e:  # pragma: no cover
        # There doesn't seem to be a way to get special errors for
        # credential issues, region issues, etc. in pyarrow (unlike s3fs).
        # So we include a blanket message to verify these details.
        raise BodoError(
            f"error from pyarrow S3FileSystem: {type(e).__name__}: {str(e)}\n{bodo_error_msg}"
        )


def s3_list_dir_fnames(fs, path):
    """
    If path is a directory, return all file names in the directory.
    This returns the base name without the path:
    ["file_name1", "file_name2", ...]
    If path is a file, return None
    """

    from pyarrow import fs as pa_fs

    file_names = None
    try:
        # check if path is a directory, and if there is a zero-size object
        # with the name of the directory. If there is, we have to omit it
        # because pq.ParquetDataset will throw Invalid Parquet file size is 0
        # bytes
        if s3_is_directory(fs, path):
            options = urlparse(path)
            # Remove the s3:// prefix if it exists (and other path sanitization)
            path_ = (options.netloc + options.path).rstrip("/")
            file_selector = pa_fs.FileSelector(path_, recursive=False)
            file_stats = fs.get_file_info(
                file_selector
            )  # this is "s3://bucket/path-to-dir"

            if (
                file_stats
                and file_stats[0].path in [path_, f"{path_}/"]
                and int(file_stats[0].size or 0)
                == 0  # FileInfo.size is None for directories, so convert to 0 before comparison
            ):  # pragma: no cover
                # excluded from coverage because haven't found a reliable way
                # to create 0 size object that is a directory. For example:
                # fs.mkdir(path) sometimes doesn't do anything at all
                # get actual names of objects inside the dir
                file_stats = file_stats[1:]
            file_names = [file_stat.base_name for file_stat in file_stats]
    except BodoError:  # pragma: no cover
        raise
    except Exception as e:  # pragma: no cover
        # There doesn't seem to be a way to get special errors for
        # credential issues, region issues, etc. in pyarrow (unlike s3fs).
        # So we include a blanket message to verify these details.
        raise BodoError(
            f"error from pyarrow S3FileSystem: {type(e).__name__}: {str(e)}\n{bodo_error_msg}"
        )

    return file_names


def hdfs_is_directory(path):
    """
    Return whether hdfs path is a directory or not
    """
    # this HadoopFileSystem is the new file system of pyarrow
    from pyarrow.fs import FileType, HadoopFileSystem

    check_java_installation(path)

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


def abfs_get_fs(storage_options: dict[str, str] | None):  # pragma: no cover
    from pyarrow.fs import AzureFileSystem

    def get_attr(opt_key: str, env_key: str) -> str | None:
        opt_val = storage_options.get(opt_key) if storage_options else None
        if (
            opt_val is not None
            and os.environ.get(env_key) is not None
            and opt_val != os.environ.get(env_key)
        ):
            warnings.warn(
                BodoWarning(
                    f"abfs_get_fs: Both {opt_key} in storage_options and {env_key} in environment variables are set. The value in storage_options will be used ({opt_val})."
                )
            )
        return opt_val or os.environ.get(env_key)

    # account_name is always required, until PyArrow or we support
    # - anonymous access
    # - parsing a connection string
    account_name = get_attr("account_name", "AZURE_STORAGE_ACCOUNT_NAME")
    # PyArrow currently only supports:
    # - Passing in Account Key Directly
    # - Default Credential Chain, i.e. ENVs, shared config, VM identity, etc.
    #   when nothing else is provided
    # To support other credential formats like SAS tokens, we need to use ENVs
    account_key = get_attr("account_key", "AZURE_STORAGE_ACCOUNT_KEY")

    if account_name is None:
        raise BodoError(
            "abfs_get_fs: Azure storage account name is not provided. Please set either the account_name in the storage_options or the AZURE_STORAGE_ACCOUNT_NAME environment variable."
        )

    # Note, Azure validates credentials at use-time instead of at
    # initialization
    return AzureFileSystem(account_name, account_key=account_key)


"""
Based on https://github.com/apache/arrow/blob/ab432b1362208696e60824b45a5599a4e91e6301/cpp/src/arrow/filesystem/azurefs.cc#L68
"""


def azure_storage_account_from_path(path: str) -> str | None:
    parsed = urlparse(path)
    host = parsed.hostname
    if host is None:
        return None

    if host.endswith(".blob.core.windows.net"):
        return host[: len(host) - len(".blob.core.windows.net")]
    if host.endswith(".dfs.core.windows.net"):
        return host[: len(host) - len(".dfs.core.windows.net")]
    return parsed.username


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


def find_file_name_or_handler(path, ftype, storage_options=None):
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
        fs = get_s3_fs_from_path(path, storage_options=storage_options)
        all_files = s3_list_dir_fnames(fs, path)  # can return None if not dir
        path_ = (parsed_url.netloc + parsed_url.path).rstrip("/")
        fname = path_

        if all_files:
            all_files = [
                (path_ + "/" + f) for f in sorted(filter(filter_func, all_files))
            ]
            # FileInfo.size is None for directories, so we convert None to 0
            # before comparison with 0
            all_csv_files = [
                f for f in all_files if int(fs.get_file_info(f).size or 0) > 0
            ]

            if len(all_csv_files) == 0:  # pragma: no cover
                # TODO: test
                raise BodoError(err_msg)
            fname = all_csv_files[0]

        f_size = int(
            fs.get_file_info(fname).size or 0
        )  # will be None for directories, so convert to 0 if that's the case

        # Arrow's S3FileSystem has some performance issues when used
        # with pandas.read_csv, which we do at compile-time.
        # Currently the issue seems related to using the output of
        # fs.open_input_file / fs.open_input_stream
        # which is a NativeFile.
        # Performance is much better (and on par with s3fs)
        # when we use an fsspec wrapper. The only difference
        # we see is that the output of fs._open is an
        # ArrowFile, which shouldn't make a difference, but it seems to.
        # We've reported the issue to
        # Pandas (https://github.com/pandas-dev/pandas/issues/46823)
        # and Arrow (https://issues.apache.org/jira/browse/ARROW-16272),
        # but in the meantime, we're using an ArrowFSWrapper for good performance.
        fs = ArrowFSWrapper(fs)
        file_name_or_handler = fs._open(fname)
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
        if parsed_url.scheme != "":
            raise BodoError(
                f"Unrecognized scheme {parsed_url.scheme}. Please refer to https://docs.bodo.ai/latest/file_io/."
            )
        is_handler = False

        if os.path.isdir(path):
            files = filter(
                filter_func, glob.glob(os.path.join(os.path.abspath(path), "*"))
            )
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


def get_s3_bucket_region(s3_filepath, parallel):
    """
    Get the region of the s3 bucket from a s3 url of type s3://<BUCKET_NAME>/<FILEPATH>.
    PyArrow's region detection only works for actual S3 buckets.
    Returns an empty string in case region cannot be determined.
    """
    from pyarrow import fs as pa_fs

    from bodo.mpi4py import MPI

    comm = MPI.COMM_WORLD

    bucket_loc = None
    if (parallel and bodo.get_rank() == 0) or not parallel:
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
    if parallel:
        bucket_loc = comm.bcast(bucket_loc)

    return bucket_loc


@numba.njit()
def get_s3_bucket_region_njit(s3_filepath, parallel):  # pragma: no cover
    """
    njit wrapper around get_s3_bucket_region
    parallel: True when called on all processes (usually runtime),
    False when called on just one process independent of the others
    (usually compile-time).
    """
    with bodo.no_warning_objmode(bucket_loc="unicode_type"):
        bucket_loc = ""
        # The parquet read path might call this function with a list of files,
        # in which case we retrieve the region of the first one. We assume
        # every file is in the same region
        if isinstance(s3_filepath, list):
            s3_filepath = s3_filepath[0]
        if s3_filepath.startswith("s3://"):
            bucket_loc = get_s3_bucket_region(s3_filepath, parallel)
    return bucket_loc


def csv_write(path_or_buf, D, filename_prefix, is_parallel=False):  # pragma: no cover
    # This is a dummy function used to allow overload.
    return None


@overload(csv_write, no_unliteral=True)
def csv_write_overload(path_or_buf, D, filename_prefix, is_parallel=False):
    def impl(path_or_buf, D, filename_prefix, is_parallel=False):  # pragma: no cover
        # Assuming that path_or_buf is a string
        bucket_region = get_s3_bucket_region_njit(path_or_buf, parallel=is_parallel)
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
            unicode_to_utf8(filename_prefix),
        )
        # Check if there was an error in the C++ code. If so, raise it.
        bodo.utils.utils.check_and_propagate_cpp_exception()

    return impl


class StorageOptionsDictType(types.Opaque):
    def __init__(self):
        super(StorageOptionsDictType, self).__init__(name="StorageOptionsDictType")


storage_options_dict_type = StorageOptionsDictType()
types.storage_options_dict_type = storage_options_dict_type  # type: ignore
register_model(StorageOptionsDictType)(models.OpaqueModel)


@unbox(StorageOptionsDictType)
def unbox_storage_options_dict_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


def get_storage_options_pyobject(storage_options):  # pragma: no cover
    pass


@overload(get_storage_options_pyobject, no_unliteral=True)
def overload_get_storage_options_pyobject(storage_options):
    """generate a pyobject for the storage_options to pass to C++"""
    storage_options_val = get_overload_constant_dict(storage_options)
    func_text = "def impl(storage_options):\n"
    func_text += "  with bodo.no_warning_objmode(storage_options_py='storage_options_dict_type'):\n"
    func_text += f"    storage_options_py = {str(storage_options_val)}\n"
    func_text += "  return storage_options_py\n"
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    return loc_vars["impl"]


class ArrowFs(types.Type):
    def __init__(self, name=""):  # pragma: no cover
        super(ArrowFs, self).__init__(name=f"ArrowFs({name})")


register_model(ArrowFs)(models.OpaqueModel)


@box(ArrowFs)
def box_ArrowFs(typ, val, c):
    fs_ptr_obj = c.pyapi.from_native_value(types.RawPointer("fs_ptr"), val)
    get_fs_obj = c.pyapi.unserialize(c.pyapi.serialize_object(get_pyarrow_fs_from_ptr))
    fs_obj = c.pyapi.call_function_objargs(get_fs_obj, [fs_ptr_obj])
    c.pyapi.decref(fs_ptr_obj)
    c.pyapi.decref(get_fs_obj)
    return fs_obj


ll.add_symbol("arrow_filesystem_del_py_entry", arrow_cpp.arrow_filesystem_del_py_entry)


@intrinsic
def _arrow_filesystem_del(typingctx, fs_instance):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [lir.LiteralStructType([lir.IntType(8).as_pointer(), lir.IntType(1)])],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="arrow_filesystem_del_py_entry"
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        builder.call(fn_tp, args)

    return types.void(types.optional(ArrowFs())), codegen


def arrow_filesystem_del(fs_instance):
    pass


@overload(arrow_filesystem_del)
def overload_arrow_filesystem_del(fs_instance):
    """Delete ArrowFs instance"""

    def impl(fs_instance):
        return _arrow_filesystem_del(fs_instance)

    return impl
