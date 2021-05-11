import asyncio
import os
import threading
from concurrent import futures

import pyarrow.parquet as pq

from bodo.io.fs_io import get_s3_bucket_region_njit

# Monkey-patching pyarrow.parquet


def get_parquet_filesnames_from_deltalake(delta_lake_path):
    """Get sorted list of parquet file names in a DeltaLake 'delta_lake_path'.
    Is meant to be called only on rank 0"""

    try:
        from deltalake import DeltaTable
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Bodo Error: please pip install the 'deltalake' package to read parquet from delta lake"
        )

    file_names = None
    path = delta_lake_path.rstrip("/")

    # The DeltaTable API doesn't have automatic S3 region detection and doesn't
    # seem to provide a way to specify a region except through the AWS_DEFAULT_REGION
    # environment variable.
    # So, we detect the region using our existing infrastructure and set the env var
    # so that it's picked up by the deltalake library
    aws_default_region_set = "AWS_DEFAULT_REGION" in os.environ  # is the env var set
    orig_aws_default_region = os.environ.get(
        "AWS_DEFAULT_REGION", ""
    )  # get original value
    aws_default_region_modified = False  # not modified yet
    if delta_lake_path.startswith("s3://"):
        # (XXX) Check that anon is False, else display error/warning?
        s3_bucket_region = get_s3_bucket_region_njit(delta_lake_path, parallel=False)
        if s3_bucket_region != "":
            os.environ["AWS_DEFAULT_REGION"] = s3_bucket_region
            aws_default_region_modified = True  # mark as modified

    dt = DeltaTable(delta_lake_path)
    file_names = dt.files()

    # pq.ParquetDataset() needs the full path for each file
    file_names = [(path + "/" + f) for f in sorted(file_names)]

    # Restore AWS_DEFAULT_REGION env var if it was modified
    if aws_default_region_modified:
        if aws_default_region_set:
            # If it was originally set to a value, restore it to that value
            os.environ["AWS_DEFAULT_REGION"] = orig_aws_default_region
        else:
            # Else delete the env var
            del os.environ["AWS_DEFAULT_REGION"]

    return file_names


def get_dataset_schema(dataset):
    # All of the code in this function is copied from
    # pyarrow.parquet.ParquetDataset.validate_schemas and is the first part
    # of that function
    if dataset.metadata is None and dataset.schema is None:
        if dataset.common_metadata is not None:  # pragma: no cover
            dataset.schema = dataset.common_metadata.schema
        else:
            dataset.schema = dataset.pieces[0].get_metadata().schema
    elif dataset.schema is None:
        dataset.schema = dataset.metadata.schema

    # Verify schemas are all compatible
    dataset_schema = dataset.schema.to_arrow_schema()
    # Exclude the partition columns from the schema, they are provided
    # by the path, not the DatasetPiece
    if dataset.partitions is not None:
        for partition_name in dataset.partitions.partition_names:
            if dataset_schema.get_field_index(partition_name) != -1:  # pragma: no cover
                field_idx = dataset_schema.get_field_index(partition_name)
                dataset_schema = dataset_schema.remove(field_idx)

    return dataset_schema


class VisitLevelThread(threading.Thread):
    """Thread that is used to traverse directory tree in ParquetManifest.
    We use a separate thread in case the main thread already
    has an event loop running (e.g. Jupyter with tornado)
    See Bodo changes to ParquetManifest below for more information.
    """

    def __init__(self, manifest):
        threading.Thread.__init__(self)
        self.manifest = manifest
        self.exc = None

    def run(self):
        try:
            manifest = self.manifest
            manifest.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(manifest.loop)
            manifest.loop.run_until_complete(
                manifest._visit_level(0, manifest.dirpath, [])
            )
        except Exception as e:
            self.exc = e
        finally:
            if hasattr(manifest, "loop") and not manifest.loop.is_closed():
                manifest.loop.close()

    def join(self):
        super(VisitLevelThread, self).join()
        if self.exc:
            raise self.exc


# We modify pyarrow.parquet.ParquetManifest to use coroutine-based concurrency
# for _visit_level tasks and thread-based concurrency for parallelism
# of IO calls, because deadlocks can occur in the original code.
# The original code uses a thread pool for _visit_level but this can lead
# to deadlock due to dependencies between threads when the thread pool is
# exhausted (new threads can't start because the pool is exhausted
# and parent threads don't finish because they are waiting on children).
# Instead, we use the thread pool only for the IO call (fs.walk()).
# A thread pool is used because we don't want an unlimited number of IO
# calls running at the same time.
# IO will be parallelized to the extent that it happens outside the GIL.
class ParquetManifest:
    def __init__(
        self,
        dirpath,
        open_file_func=None,
        filesystem=None,
        pathsep="/",
        partition_scheme="hive",
        metadata_nthreads=1,
    ):
        filesystem, dirpath = pq._get_filesystem_and_path(filesystem, dirpath)
        self.filesystem = filesystem
        self.open_file_func = open_file_func
        self.pathsep = pathsep
        self.dirpath = pq._stringify_path(dirpath)
        self.partition_scheme = partition_scheme
        self.partitions = pq.ParquetPartitions()
        self.pieces = []
        self._metadata_nthreads = metadata_nthreads
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=metadata_nthreads)

        self.common_metadata_path = None
        self.metadata_path = None

        # Bodo change: filter for Delta Lake (only consider files in Delta table)
        self.delta_lake_filter = set()

        # Bodo change (run _visit_level with asyncio)
        # Do traversal in a separate thread in case the main thread already
        # has an event loop running (e.g. Jupyter with tornado)
        thread = VisitLevelThread(self)
        thread.start()
        thread.join()

        # Due to concurrency, pieces will potentially by out of order if the
        # dataset is partitioned so we sort them to yield stable results
        self.pieces.sort(key=lambda piece: piece.path)

        if self.common_metadata_path is None:
            # _common_metadata is a subset of _metadata
            self.common_metadata_path = self.metadata_path

        self._thread_pool.shutdown()

    # Bodo change (add async to method)
    async def _visit_level(self, level, base_path, part_keys):
        fs = self.filesystem

        # Bodo change (do fs.walk with thread pool executor)
        _, directories, files = await self.loop.run_in_executor(
            self._thread_pool,
            lambda fs, base_bath: next(fs.walk(base_path)),
            fs,
            base_path,
        )

        # Bodo change
        if level == 0 and "_delta_log" in directories:
            # this is a Delta Lake table
            self.delta_lake_filter = set(
                get_parquet_filesnames_from_deltalake(base_path)
            )

        filtered_files = []
        for path in files:
            if path == "":  # Bodo change
                continue
            full_path = self.pathsep.join((base_path, path))
            if path.endswith("_common_metadata"):
                self.common_metadata_path = full_path
            elif path.endswith("_metadata"):
                self.metadata_path = full_path
            elif self._should_silently_exclude(path):
                continue
            elif self.delta_lake_filter and full_path not in self.delta_lake_filter:
                continue
            else:
                filtered_files.append(full_path)

        # ARROW-1079: Filter out "private" directories starting with underscore
        filtered_directories = [
            self.pathsep.join((base_path, x))
            for x in directories
            if not pq._is_private_directory(x)
        ]

        filtered_files.sort()
        filtered_directories.sort()

        if len(filtered_files) > 0 and len(filtered_directories) > 0:  # pragma: no cover
            raise ValueError(
                "Found files in an intermediate " "directory: {}".format(base_path)
            )
        elif len(filtered_directories) > 0:
            # Bodo change (await)
            await self._visit_directories(level, filtered_directories, part_keys)
        else:
            self._push_pieces(filtered_files, part_keys)

    # Bodo change (add async to method)
    async def _visit_directories(self, level, directories, part_keys):
        aws = []  # awaitables
        for path in directories:
            head, tail = pq._path_split(path, self.pathsep)
            name, key = pq._parse_hive_partition(tail)

            index = self.partitions.get_index(level, name, key)
            dir_part_keys = part_keys + [(name, index)]
            # Bodo change: always spawn _visit_level as a coroutine, remove
            # condition that decides whether to run in separate thread
            aws.append(self._visit_level(level + 1, path, dir_part_keys))
        # Bodo change (run _visit_level with asyncio)
        await asyncio.wait(aws)


# Get the methods that we are not monkey-patching from pyarrow.parquet
ParquetManifest._should_silently_exclude = pq.ParquetManifest._should_silently_exclude
ParquetManifest._parse_partition = pq.ParquetManifest._parse_partition
ParquetManifest._push_pieces = pq.ParquetManifest._push_pieces
# Replace pyarrow.parquet.ParquetManifest with ours
pq.ParquetManifest = ParquetManifest
