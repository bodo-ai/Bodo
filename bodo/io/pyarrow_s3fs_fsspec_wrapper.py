from urllib.parse import urlparse

import pyarrow.fs as pa_fs
from fsspec import AbstractFileSystem
from pyarrow.fs import S3FileSystem


class PyArrowS3FS(AbstractFileSystem):
    """
    Minimal FSSpec wrapper around pyarrow.fs.S3FileSystem
    Based on:
    https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/zip.html#ZipFileSystem
    https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/hdfs.html#PyArrowHDFS
    https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/dask.html#DaskWorkerFileSystem
    https://filesystem-spec.readthedocs.io/en/latest/api.html

    This is not a complete fsspec implementation and is missing various required APIs.
    We use this wrapper to be able to use it at compile-time, specifically to pass it to
    pyarrow.pq.ParquetDataset. If a pyarrow.fs.S3FileSystem instance is passed in directly,
    it uses pyarrow.pq.ParquetDatasetV2 which doesn't have all the functionality for
    filter push-down that is required at compile-time. But using this wrapper class and
    setting use_legacy_dataset=True allows us to continue using pyarrow.pq.ParquetDataset.
    """

    protocol = "s3"

    def __init__(
        self,
        *,
        access_key=None,
        secret_key=None,
        session_token=None,
        anonymous=False,
        region=None,
        scheme=None,
        endpoint_override=None,
        background_writes=True,
        role_arn=None,
        session_name=None,
        external_id=None,
        load_frequency=900,
        proxy_options=None,
        **kwargs,
    ):
        super().__init__(self, **kwargs)
        self.pa_s3fs = S3FileSystem(
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            anonymous=anonymous,
            region=region,
            scheme=scheme,
            endpoint_override=endpoint_override,
            background_writes=background_writes,
            role_arn=role_arn,
            session_name=session_name,
            external_id=external_id,
            load_frequency=load_frequency,
            proxy_options=proxy_options,
        )

    def __getattribute__(self, name: str):

        if name == "__class__":
            return PyArrowS3FS

        # Functions defined in this class
        if name in [
            "__init__",
            "__getattribute__",
            "_open",
            "open",
            "ls",
            "isdir",
            "isfile",
        ]:
            return lambda *args, **kw: getattr(PyArrowS3FS, name)(self, *args, **kw)

        d = object.__getattribute__(self, "__dict__")
        pa_s3fs = d.get("pa_s3fs", None)  # fs is not immediately defined
        if name == "pa_s3fs":
            return pa_s3fs

        # If a function/attribute defined in pyarrow.fs.S3FileSystem:
        if pa_s3fs is not None and hasattr(pa_s3fs, name):
            return getattr(pa_s3fs, name)

        return super().__getattribute__(name)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        # Remove the s3:// prefix if it exists
        options = urlparse(path)
        path_ = options.netloc + options.path
        return self.pa_s3fs.open_input_file(path_)

    # Adding a pragma: no cover since coverage collection doesn't seem to be picking
    # up the usage
    def ls(self, path, detail=True, **kwargs):  # pragma: no cover
        """
        Similar to bodo.io.fs_io.s3_list_dir_fnames.
        When s3fs.ls (s3fs is an fsspec impl) is called on a directory (not the top-level),
        the output list contains an entry for the directory as well. We don't include this
        entry. When called on a file, we return a list with a singular entry for the file,
        similar to s3fs.
        """

        options = urlparse(path)
        # Remove the s3:// prefix if it exists (and other path sanitization)
        path_ = (options.netloc + options.path).rstrip("/")

        file_selector = pa_fs.FileSelector(path_, recursive=False)
        file_stats = self.pa_s3fs.get_file_info(file_selector)
        if len(file_stats) == 0:
            if self.isfile(path):
                if detail:
                    return [{"type": "file", "name": path_}]
                else:
                    return [path_]
            return []  # Probably a non-existent path

        # Remove the directory object itself if it appears in the list
        if (
            file_stats
            and file_stats[0].path in [path_, f"{path_}/"]
            and int(file_stats[0].size or 0) == 0
        ):
            file_stats = file_stats[1:]

        out = []
        if detail:
            for file_stat in file_stats:
                p = {}
                if file_stat.type == pa_fs.FileType.Directory:
                    p["type"] = "directory"
                elif file_stat.type == pa_fs.FileType.File:
                    p["type"] = "file"
                else:
                    p["type"] = "unknown"  # XXX Wasn't sure what to put here
                p["name"] = file_stat.base_name
                out.append(p)
        else:
            out = [file_stat.base_name for file_stat in file_stats]
        return out

    def isdir(self, path):
        """
        Similar to bodo.io.fs_io.s3_is_directory
        """
        options = urlparse(path)
        # Remove the s3:// prefix if it exists (and other path sanitization)
        path_ = (options.netloc + options.path).rstrip("/")
        path_info = self.pa_s3fs.get_file_info(path_)
        return (not path_info.size) and (path_info.type == pa_fs.FileType.Directory)

    def isfile(self, path):
        options = urlparse(path)
        # Remove the s3:// prefix if it exists (and other path sanitization)
        path_ = (options.netloc + options.path).rstrip("/")
        path_info = self.pa_s3fs.get_file_info(path_)
        return path_info.type == pa_fs.FileType.File
