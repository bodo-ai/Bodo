# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
S3 & Hadoop file system supports
"""
from urllib.parse import urlparse


def get_s3_fs():
    """
    initialize S3FileSystem with credentials
    """
    try:
        import s3fs
    except: # pragma: no cover
        raise BodoError("Reading from s3 requires s3fs currently.")

    import os

    custom_endpoint = os.environ.get("AWS_S3_ENDPOINT", None)
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

    # always use s3fs.S3FileSystem.clear_instance_cache()
    # before initializing S3FileSystem due to inconsistent file system
    # between to_parquet to read_parquet
    if custom_endpoint is not None and (
        aws_access_key_id is None or aws_secret_access_key is None
    ): # pragma: no cover
        warnings.warn(
            BodoWarning(
                "Reading from s3 with custom_endpoint, "
                "but environment variables AWS_ACCESS_KEY_ID or "
                "AWS_SECRET_ACCESS_KEY is not set."
            )
        )
    s3fs.S3FileSystem.clear_instance_cache()
    fs = s3fs.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        client_kwargs={"endpoint_url": custom_endpoint},
    )

    return fs


# hdfs related functions(hdfs_list_dir_fnames) should be included in 
# coverage once hdfs tests are included in CI
def get_hdfs_fs(path): # pragma: no cover
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
    path = options.path
    # creates a new Hadoop file system from uri
    try:
        fs = HdFS(host=options.hostname, port=options.port, user=options.username)
    except Exception as e:
        raise BodoError("Hadoop file system cannot be created: {}".format(e))

    return fs


def s3_list_dir_fnames(fs, path):
    """
    If path is a directory, return all file names in the directory:
    ["s3://bucket-name/path/file_name", ...]
    If path is a file, return None
    """
    file_names = None
    try:
        # check if path is a directory, and if there is a zero-size object
        # with the name of the directory. If there is, we have to omit it
        # because pq.ParquetDataset will throw Invalid Parquet file size is 0
        # bytes
        path_info = fs.info(path)
        if (
            path_info["Size"] == 0 and path_info["type"] == "directory"
        ):  # pragma: no cover
            # excluded from coverage because haven't found a reliable way
            # to create 0 size object that is a directory. For example:
            # fs.mkdir(path)  sometimes doesn't do anything at all
            files = fs.ls(path)  # this is "s3://bucket/path-to-dir"
            if (
                files
                and (files[0] == path[5:] or files[0] == path[5:] + "/")
                and fs.info("s3://" + files[0])["Size"] == 0
            ):
                # get actual names of objects inside the dir
                file_names = ["s3://" + fname for fname in files[1:]]
            else:
                file_names = ["s3://" + fname for fname in files]

    except:  # pragma: no cover
        pass

    return file_names


def hdfs_list_dir_fnames(path):  # pragma: no cover
    """
    initialize pyarrow.fs.HadoopFileSystem from path
    If path is a directory, file_names = ["hfsd://host:port/path/file_name", ...]
    If path is a file, file_names = None
    return (pyarrow.fs.HadoopFileSystem, file_names)
    """

    # this HadoopFileSystem is the new file system of pyarrow
    from pyarrow.fs import HadoopFileSystem, FileSelector, FileType

    file_names = None
    options = urlparse(path)
    hdfs_path = options.path  # path within hdfs(i.e. dir/file)

    try:
        hdfs = HadoopFileSystem.from_uri(path)
    except:
        raise BodoError(" Hadoop file system cannot be created: {}".format(e))
    # prefix in form of hdfs://host:port
    prefix = path[: len(path) - len(hdfs_path)]
    # target stat of the path: file or just the directory itself
    target_stat = hdfs.get_file_info([path])

    if target_stat[0].type in (FileType.NotFound, FileType.Unknown):
        raise BodoError("{} is a " "non-existing or unreachable file".format(path))

    if (not target_stat[0].size) and target_stat[0].type == FileType.Directory:
        file_selector = FileSelector(hdfs_path, recursive=True)
        try:
            file_stats = hdfs.get_file_info(file_selector)
        except Exception as e:
            raise BodoError(
                "Exception on getting directory info " "of {}: {}".format(hdfs_path, e)
            )
        for file_stat in file_stats:
            file_names = [prefix + file_stat.path for file_stat in file_stats]

    return (hdfs, file_names)
