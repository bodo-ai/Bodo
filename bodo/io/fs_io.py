"""
S3 & Hadoop file system supports, and file system dependent calls.
This file should import JIT lazily to avoid slowing down non-JIT code paths.
"""

from __future__ import annotations

import os
import typing as pt
import warnings
from dataclasses import dataclass
from glob import has_magic
from urllib.parse import ParseResult, urlparse

import pyarrow as pa
from fsspec.implementations.arrow import (
    ArrowFile,
    ArrowFSWrapper,
    wrap_exceptions,
)
from pyarrow.fs import FSSpecHandler, PyFileSystem

import bodo
from bodo import BodoWarning

# Same as _fs_io.cpp
GCS_RETRY_LIMIT_SECONDS = 2


@dataclass
class AWSCredentials:
    access_key: str
    secret_key: str
    session_token: str | None = None
    region: str | None = None


# ----- monkey-patch fsspec.implementations.arrow.ArrowFSWrapper._open --------
def fsspec_arrowfswrapper__open(self, path, mode="rb", block_size=None, **kwargs):
    if mode == "rb":
        try:  # Bodo change: try to open the file for random access first
            # We need random access to read parquet file metadata
            stream = self.fs.open_input_file(path)
        except Exception:  # pragma: no cover
            stream = self.fs.open_input_stream(path)
    elif mode == "wb":  # pragma: no cover
        stream = self.fs.open_output_stream(path)
    else:
        raise ValueError(f"unsupported mode for Arrow filesystem: {mode!r}")

    return ArrowFile(self, stream, path, mode, block_size, **kwargs)


ArrowFSWrapper._open = wrap_exceptions(fsspec_arrowfswrapper__open)
# -----------------------------------------------------------------------------


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
        raise ValueError(
            "Couldn't import s3fs, which is required for certain types of S3 access."
            " s3fs can be installed by calling"
            " 'conda install -c conda-forge s3fs'.\n"
        )


def validate_huggingface_hub_installed():
    """
    Validate that huggingface_hub is installed. Raise an error if not.
    """
    try:
        import huggingface_hub  # noqa
    except ImportError:
        raise ValueError(
            "Cannot import huggingface_hub, which is required for reading from Hugging Face."
            " Please make sure the huggingface_hub package is installed."
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
    region = get_s3_bucket_region_wrapper(path, parallel=parallel)
    if region == "":
        region = None
    return get_s3_fs(region, storage_options)


def get_gcs_fs(path, storage_options=None):
    """Get PyArrow GcsFileSystem object to read GCS path. Tries accessing the path
    with anonymous=True if not authenticated since Arrow doesn't try automatically.
    """
    import datetime

    from pyarrow.fs import GcsFileSystem

    # PyArrow seems to hang for a long time if retry isn't set
    options = {"retry_time_limit": datetime.timedelta(seconds=GCS_RETRY_LIMIT_SECONDS)}

    anon = False
    if storage_options:
        anon = storage_options.pop("anon", anon)
        anon = storage_options.pop("anonymous", anon)
        options.update(storage_options)

    fs = GcsFileSystem(anonymous=anon, **options)

    if anon:
        return fs

    # Try with anonymous=True if not authenticated since Arrow doesn't try automatically
    try:
        parsed_url = urlparse(path)
        path_ = (parsed_url.netloc + parsed_url.path).rstrip("/")
        fs.get_file_info(path_)
        return fs
    except OSError as e:
        if "Could not create a OAuth2 access token to authenticate the request" in str(
            e
        ):
            return GcsFileSystem(anonymous=True, **options)
        raise e


def get_hf_fs(storage_options=None):
    """Create an Arrow file system object for reading Hugging Face datasets."""
    validate_huggingface_hub_installed()
    import huggingface_hub
    from pyarrow.fs import FSSpecHandler, PyFileSystem

    options = {}
    if storage_options:
        options.update(storage_options)

    fs = huggingface_hub.HfFileSystem(**options)
    return PyFileSystem(FSSpecHandler(fs))


# hdfs related functions should be included in
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
        raise ValueError(f"Hadoop file system cannot be created: {e}")

    return fs


def pa_fs_is_directory(fs, path):
    """
    Return whether a path (S3, GCS, ...) is a directory or not
    """
    from pyarrow import fs as pa_fs

    try:
        path_info = fs.get_file_info(path)
        if path_info.type in (pa_fs.FileType.NotFound, pa_fs.FileType.Unknown):
            raise FileNotFoundError(f"{path} is a non-existing or unreachable file")
        if (not path_info.size) and path_info.type == pa_fs.FileType.Directory:
            return True
        return False
    except (FileNotFoundError, OSError):
        raise
    except ValueError:  # pragma: no cover
        raise
    except Exception as e:  # pragma: no cover
        # There doesn't seem to be a way to get special errors for
        # credential issues, region issues, etc. in pyarrow (unlike s3fs).
        # So we include a blanket message to verify these details.
        raise ValueError(
            f"error from pyarrow FileSystem: {type(e).__name__}: {str(e)}\n{bodo_error_msg}"
        )


def pa_fs_list_dir_fnames(fs, path):
    """
    If path is a directory, return all file names in the directory.
    This returns the base name without the path:
    ["file_name1", "file_name2", ...]
    If path is a file, return None
    """

    from pyarrow import fs as pa_fs

    path = path if isinstance(path, list) else [path]

    file_names = []
    for p in path:
        try:
            if pa_fs_is_directory(fs, p):
                file_selector = pa_fs.FileSelector(p, recursive=True)
                file_stats = fs.get_file_info(file_selector)
                file_names += [
                    file_stat.path
                    for file_stat in file_stats
                    if file_stat.type != pa_fs.FileType.Directory
                ]
            else:
                file_names.append(p)
        except ValueError:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            # There doesn't seem to be a way to get special errors for
            # credential issues, region issues, etc. in pyarrow (unlike s3fs).
            # So we include a blanket message to verify these details.
            raise ValueError(
                f"error from pyarrow FileSystem: {type(e).__name__}: {str(e)}\n{bodo_error_msg}"
            )

    return file_names


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
        raise ValueError(
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


def expand_glob(protocol: str, fs: pa.fs.FileSystem | None, path: str) -> list[str]:
    """
    Return a list of path names that match glob pattern
    given by path.

    Args:
        protocol (str): Protocol for the path. e.g.
            "" -> local
            "s3" -> S3
        fs (pa.fs.FileSystem | None): Filesystem to use
            for getting list of files. This can be None
            in the local filesystem case, i.e. protocol
            is an empty string.
        path (str): Glob pattern.

    Returns:
        list[str]: List of files returned by expanding the glob
            pattern.
    """
    if not protocol and fs is None:
        from fsspec.implementations.local import LocalFileSystem

        fs = LocalFileSystem()
    elif fs is None:
        raise ValueError(
            f"glob: 'fs' cannot be None in the non-local ({protocol}) filesystem case!"
        )

    if isinstance(fs, pa.fs.FileSystem):
        from fsspec.implementations.arrow import ArrowFSWrapper

        fs = ArrowFSWrapper(fs)

    try:
        # Arrow's FileSystem.glob() doesn't support Windows backslashes
        files = fs.glob(path.replace("\\", "/"))
    except Exception:  # pragma: no cover
        raise ValueError(f"glob pattern expansion not supported for {protocol}")

    return files


def expand_path_globs(fpath: str | list[str], protocol: str, fs) -> list[str]:
    """Expand any glob strings in the provided file path(s).

    Args:
        fpath (str | list[str]): file paths or list of file paths.
        protocol (str): protocol for filesystem, e.g. "s3", "gcs", and "" for local.
        fs (pa.fs.FileSystem): filesystem object

    Raises:
        ValueError: error if no files found matching glob pattern

    Returns:
        list[str]: expanded list of paths
    """

    new_fpath = fpath

    if isinstance(fpath, list):
        # Expand any glob strings in the list in order to generate a
        # single list of fully realized paths to parquet files.
        # For example: ["A/a.pq", "B/*.pq"] might expand to
        # ["A/a.pq", "B/part-0.pq", "B/part-1.pq"]
        new_fpath = []
        for p in fpath:
            if has_magic(p):
                new_fpath += expand_glob(protocol, fs, p)
            else:
                new_fpath.append(p)
        if len(new_fpath) == 0:
            raise ValueError("No files found matching glob pattern")

    elif has_magic(fpath):
        new_fpath = expand_glob(protocol, fs, fpath)
        if len(new_fpath) == 0:
            raise ValueError("No files found matching glob pattern")

    return new_fpath


def getfs(
    fpath: str | list[str],
    protocol: str,
    storage_options: dict[str, pt.Any] | None = None,
    parallel: bool = False,
) -> PyFileSystem | pa.fs.FileSystem:
    """
    Get filesystem for the provided file path(s).

    Args:
        fpath (str | list[str]): Filename or list of filenames.
        protocol (str): Protocol for the filesystem. e.g. "" (for local), "s3", etc.
        storage_options (Optional[dict], optional): Optional storage_options to
            use when building the filesystem. Only supported in the S3 case
            at this time. Defaults to None.
        parallel (bool, optional): Whether this function is being called in parallel.
            Defaults to False.

    Returns:
        Filesystem implementation. This is either a PyFileSystem wrapper over
        s3fs or a native PyArrow filesystem.
    """
    # NOTE: add remote filesystems to REMOTE_FILESYSTEMS
    if (
        protocol in ("s3", "s3a")
        and storage_options
        and ("anon" not in storage_options or len(storage_options) > 1)
    ):
        # "anon" is the only storage_options supported by PyArrow
        # If other storage_options fields are given,
        # we need to use S3Fs to read instead
        validate_s3fs_installed()
        import s3fs

        sopts = storage_options.copy()
        if "AWS_S3_ENDPOINT" in os.environ and "endpoint_url" not in sopts:
            sopts["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
        s3_fs = s3fs.S3FileSystem(
            **sopts,
        )
        return PyFileSystem(FSSpecHandler(s3_fs))
    if protocol in ("s3", "s3a"):
        return (
            get_s3_fs_from_path(
                fpath,
                parallel=parallel,
                storage_options=storage_options,
            )
            if not isinstance(fpath, list)
            else get_s3_fs_from_path(
                fpath[0],
                parallel=parallel,
                storage_options=storage_options,
            )
        )
    if storage_options is not None and len(storage_options) > 0:
        raise ValueError(
            f"ParquetReader: `storage_options` is not supported for protocol {protocol}"
        )

    if protocol in {"gcs", "gs"}:
        return get_gcs_fs(fpath, storage_options=storage_options)
    elif protocol == "http":
        import fsspec

        return PyFileSystem(FSSpecHandler(fsspec.filesystem("http")))
    elif protocol in {"abfs", "abfss"}:  # pragma: no cover
        if not storage_options:
            storage_options = {}
        if "account_name" not in storage_options:
            # Extract the storage account from the path, assumes all files are in the same storage account
            account_name = azure_storage_account_from_path(
                fpath if not isinstance(fpath, list) else fpath[0]
            )
            if account_name is not None:
                storage_options["account_name"] = account_name

        return abfs_get_fs(storage_options)
    elif protocol == "hdfs":  # pragma: no cover
        return (
            get_hdfs_fs(fpath) if not isinstance(fpath, list) else get_hdfs_fs(fpath[0])
        )
    # HuggingFace datasets
    elif protocol == "hf":
        return get_hf_fs(storage_options)
    else:
        return pa.fs.LocalFileSystem()


def get_uri_scheme(path):
    """Get URI scheme from path (e.g. "s3", "gcs").
    Used in C++ code to avoid including extra dependencies like boost::url.
    """
    parsed_url: ParseResult = urlparse(path)
    return parsed_url.scheme


@pt.overload
def parse_fpath(fpath: str) -> tuple[str, ParseResult, str]: ...


@pt.overload
def parse_fpath(fpath: list[str]) -> tuple[list[str], ParseResult, str]: ...


def parse_fpath(fpath: str | list[str]) -> tuple[str | list[str], ParseResult, str]:
    """
    Parse a filepath and extract properties such as the relevant
    protocol, scheme, netloc, etc.
    In case it's a list of filepaths, this validates that properties
    such as the protocol and netloc (i.e. the bucket in the S3 case)
    are the same for all filepaths in the list.

    Args:
        fpath (str | list[str]): Filepath or list of filepaths to parse.

    Returns:
        tuple[str | list[str], ParseResult, str]:
            - str: Sanitized version of the filepath(s).
            - ParseResult: ParseResult object containing the
                scheme, netloc, etc.
            - str: The protocol associate with the filepath(s).
                e.g. "" (local), "s3" (S3), etc.
    """

    if isinstance(fpath, list):
        # list of file paths
        parsed_url = urlparse(fpath[0])
        protocol = parsed_url.scheme
        bucket_name = parsed_url.netloc  # netloc can be empty string (e.g. non s3)
        for i in range(len(fpath)):
            f = fpath[i]
            u_p = urlparse(f)
            # make sure protocol and bucket name of every file matches
            if u_p.scheme != protocol:
                raise ValueError(
                    "All parquet files must use the same filesystem protocol"
                )
            if u_p.netloc != bucket_name:
                raise ValueError("All parquet files must be in the same S3 bucket")
            fpath[i] = f.rstrip("/")
    else:
        parsed_url: ParseResult = urlparse(fpath)
        protocol = parsed_url.scheme
        fpath = fpath.rstrip("/")

    return fpath, parsed_url, protocol


def directory_of_files_common_filter(fname):
    # Ignore the same files as pyarrow,
    # https://github.com/apache/arrow/blob/4beb514d071c9beec69b8917b5265e77ade22fb3/python/pyarrow/parquet.py#L1039
    return not (
        fname.endswith(".crc")  # Checksums
        or fname.endswith("_$folder$")  # HDFS directories in S3
        or (
            fname.startswith(".")
            and not fname.startswith("./")
            and not fname.startswith("../")
            and not fname.startswith(".\\")
            and not fname.startswith("..\\\\")
        )  # Hidden files starting with .
        or fname.startswith("_")
        and fname != "_delta_log"  # Hidden files starting with _ skip deltalake
    )


def get_compression_from_file_name(fname: str):
    """Get compression scheme from file name"""

    compression = None

    if fname.endswith(".gz"):
        compression = "gzip"
    elif fname.endswith(".bz2"):
        compression = "bz2"
    elif fname.endswith(".zip"):
        compression = "zip"
    elif fname.endswith(".xz"):
        compression = "xz"
    elif fname.endswith(".zst"):
        compression = "zstd"

    return compression


def get_all_csv_json_data_files(
    fs: pa.fs.FileSystem,
    path: str,
    protocol: str,
    parsed_url: ParseResult,
    err_msg: str,
) -> list[str]:
    """Get all data files to CSV/JSON read from path. Handles glob patterns, directories
    (including nested directories), and single files. Filters out some metadata files
    as well.

    Args:
        fs (pa.fs.FileSystem): file system to use for storage operations
        path (str): input path
        protocol (str): protocol for remote access (e.g. "s3", "gcs", "hf")
        parsed_url (ParseResult): parsed URL object for path
        err_msg (str): error message to raise if no data files are found

    Raises:
        ValueError: error when there are no data files found

    Returns:
        list[str]: list of files to read
    """
    from bodo.io.parquet_pio import get_fpath_without_protocol_prefix

    fpath_noprefix, _ = get_fpath_without_protocol_prefix(path, protocol, parsed_url)

    fpath_noprefix = expand_path_globs(fpath_noprefix, protocol, fs)

    all_files = pa_fs_list_dir_fnames(fs, fpath_noprefix)

    all_files = sorted(filter(directory_of_files_common_filter, all_files))
    # FileInfo.size is None for directories, so we convert None to 0
    # before comparison with 0
    all_data_files = [f for f in all_files if int(fs.get_file_info(f).size or 0) > 0]

    if len(all_data_files) == 0:  # pragma: no cover
        # TODO: test
        raise ValueError(err_msg)

    return all_data_files


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
        (is_handler, file_name_or_handler, fs)
        is_handler: True if file_name_or_handler is a handler,
                    False otherwise(file_name_or_handler is a file_name)
        file_name_or_handler: file_name or handler to pass to pd.read_csv()/pd.read_json()
        compression: compression scheme inferred from file name
        fs: file system for s3/hdfs
    """

    fname = path
    fs = None
    func_name = "read_json" if ftype == "json" else "read_csv"
    err_msg = f"pd.{func_name}(): there is no {ftype} file in directory: {fname}"

    path, parsed_url, protocol = parse_fpath(path)

    fs = getfs(path, protocol, storage_options=storage_options)

    all_data_files = get_all_csv_json_data_files(
        fs, path, protocol, parsed_url, err_msg
    )
    fname = all_data_files[0]

    if protocol == "":
        file_name_or_handler = fname
        is_handler = False
    else:
        file_name_or_handler = fs.open_input_file(fname)
        is_handler = True

    compression = get_compression_from_file_name(fname)

    # although fs is never used, we need to return it so that s3/hdfs
    # connections are not closed
    return is_handler, file_name_or_handler, compression, fs


def get_s3_bucket_region(s3_filepath, parallel):
    """
    Get the region of the s3 bucket from a s3 url of type s3://<BUCKET_NAME>/<FILEPATH>.
    PyArrow's region detection only works for actual S3 buckets.
    Returns an empty string in case region cannot be determined.
    """
    from mpi4py import MPI
    from pyarrow import fs as pa_fs

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


def get_s3_bucket_region_wrapper(s3_filepath, parallel):  # pragma: no cover
    """
    Wrapper around get_s3_bucket_region that handles list input and non-S3 paths.
    parallel: True when called on all processes (usually runtime),
    False when called on just one process independent of the others
    (usually compile-time).
    """
    bucket_loc = ""
    # The parquet read path might call this function with a list of files,
    # in which case we retrieve the region of the first one. We assume
    # every file is in the same region
    if isinstance(s3_filepath, list):
        s3_filepath = s3_filepath[0]
    if s3_filepath.startswith("s3://") or s3_filepath.startswith("s3a://"):
        bucket_loc = get_s3_bucket_region(s3_filepath, parallel)
    return bucket_loc
