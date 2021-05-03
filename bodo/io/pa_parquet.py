import asyncio
from concurrent import futures

import pyarrow.parquet as pq

# Monkey-patching pyarrow.parquet

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

        # Bodo change (run _visit_level with asyncio)
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._visit_level(0, self.dirpath, []))

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

        if len(filtered_files) > 0 and len(filtered_directories) > 0:
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
