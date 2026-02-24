"""Parquet I/O utilities. This file should import JIT lazily to avoid slowing down
non-JIT code paths.
"""

from __future__ import annotations

import json
import os
import random
import time
import typing as pt
import warnings
from collections import defaultdict
from typing import Any
from urllib.parse import ParseResult, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from mpi4py import MPI

import bodo
import bodo.utils.tracing as tracing
from bodo.io.fs_io import (
    expand_path_globs,
    getfs,
    parse_fpath,
)

if pt.TYPE_CHECKING:
    import pyarrow.compute as pc

    from bodo.utils.typing import FileSchema


REMOTE_FILESYSTEMS = {"s3", "gcs", "gs", "http", "hdfs", "abfs", "abfss"}
# the ratio of total_uncompressed_size of a Parquet string column vs number of values,
# below which we read as dictionary-encoded string array
READ_STR_AS_DICT_THRESHOLD = 1.0

LIST_OF_FILES_ERROR_MSG = ". Make sure the list/glob passed to read_parquet() only contains paths to files (no directories)"


def unify_schemas(
    schemas: pt.Iterable[pa.Schema],
    promote_options: pt.Literal["default", "permissive"] = "default",
) -> pa.Schema:
    """
    Same as pyarrow.unify_schemas with the difference
    that we unify `large_string` and `string` types to `string`,
    (Arrow considers them incompatible).
    Note that large_strings is not a property of parquet, but
    rather a decision made by Arrow on how to store string data
    in memory. For Bodo, we can have Arrow always read as regular
    strings and convert to Bodo's representation during read.
    Similarly, we also unify `large_binary` and `binary` to `binary`.
    We also convert LargeListType to regular list type (the type inside
    is not modified).
    Additionally, pa.list_(pa.large_string()) is converted to
    pa.list_(pa.string()).
    """
    # first replace large_string with string in every schema
    new_schemas = []
    for schema in schemas:
        for i in range(len(schema)):
            f = schema.field(i)
            if f.type == pa.large_string():
                schema = schema.set(i, f.with_type(pa.string()))
            elif f.type == pa.large_binary():
                schema = schema.set(i, f.with_type(pa.binary()))
            elif isinstance(
                f.type, (pa.ListType, pa.LargeListType)
            ) and f.type.value_type in (pa.string(), pa.large_string()):
                # This handles the pa.list_(pa.large_string()) case
                # that the next `elif` doesn't.
                schema = schema.set(
                    i,
                    f.with_type(
                        # We want to retain the name (e.g. 'element'), so we pass
                        # in a field to pa.list_ instead of a simple string type
                        # which would use 'item' by default.
                        pa.list_(pa.field(f.type.value_field.name, pa.string()))
                    ),
                )
            elif isinstance(f.type, pa.LargeListType):
                schema = schema.set(
                    i,
                    f.with_type(
                        # We want to retain the name (e.g. 'element'), so we pass
                        # in a field to pa.list_ instead of a simple string type
                        # which would use 'item' by default.
                        pa.list_(pa.field(f.type.value_field.name, f.type.value_type))
                    ),
                )
            # TODO handle arbitrary nested types
        new_schemas.append(schema)
    # now we run Arrow's regular schema unification
    return pa.unify_schemas(new_schemas, promote_options=promote_options)


class ParquetDataset:
    """Stores information about parquet dataset that is needed at compile time
    and runtime (to read the dataset). Stores the list of fragments
    (pieces) that form the dataset and filesystem object to read them.
    All of this is obtained at rank 0 using Arrow's pq.ParquetDataset() API
    (ParquetDatasetV2) and this object is broadcasted to all ranks.
    """

    def __init__(self, pa_pq_dataset, prefix=""):
        # We always get exact row counts (after filters) for now.
        self.row_level = True
        self.schema: pa.Schema = pa_pq_dataset.schema  # Arrow schema
        # We don't store the filesystem initially, and instead set it after
        # ParquetDataset is broadcasted. This is because some filesystems
        # might have pickling issues, and also because of the extra cost of
        # creating the filesystem during unpickling. Instead, all ranks
        # initialize the filesystem in parallel at the same time and only once
        self.filesystem = None
        # total number of rows in the dataset (after applying filters). This
        # is computed at runtime in `get_parquet_dataset`
        self._bodo_total_rows = 0
        # prefix that needs to be added to paths of parquet pieces to get the
        # full path to the file
        self._prefix = prefix
        # XXX pa_pq_dataset.partitioning can't be pickled, so we reconstruct
        # manually after broadcasting the dataset (see __setstate__ below)
        self.partitioning = None
        partitioning = pa_pq_dataset.partitioning
        # For some datasets, partitioning.schema contains the
        # full schema of the dataset when there aren't any partition columns
        # (bug in Arrow?) so to know if there are partition columns we also
        # need to check that the partitioning schema is not equal to the
        # full dataset schema.
        # XXX is there a better way to get partition column names?
        self.partition_names = (
            []
            if partitioning is None or partitioning.schema == pa_pq_dataset.schema
            else list(partitioning.schema.names)
        )
        # partitioning_dictionaries is an Arrow array containing the
        # partition values
        if self.partition_names:
            self.partitioning_dictionaries = partitioning.dictionaries
            self.partitioning_cls = partitioning.__class__
            self.partitioning_schema = partitioning.schema
        else:
            self.partitioning_dictionaries = {}
        # Convert large_string Arrow types to string and dictionary to int32 indices
        # (see comment in bodo.io.parquet_pio.unify_schemas)
        for i in range(len(self.schema)):
            f = self.schema.field(i)
            if f.type == pa.large_string():
                self.schema = self.schema.set(i, f.with_type(pa.string()))
            elif isinstance(f.type, pa.DictionaryType):
                if f.type.index_type != pa.int32():
                    self.schema = self.schema.set(
                        i,
                        f.with_type(
                            pa.dictionary(pa.int32(), f.type.value_type, f.type.ordered)
                        ),
                    )
        # IMPORTANT: only include partition columns in filters passed to
        # pq.ParquetDataset(), otherwise `get_fragments` could look inside the
        # parquet files
        self.pieces = [
            ParquetPiece(frag, partitioning, self.partition_names)
            for frag in pa_pq_dataset._dataset.get_fragments(
                filter=pa_pq_dataset._filter_expression
            )
        ]

    def set_fs(self, fs):
        """Set filesystem (to read fragments)"""
        self.filesystem = fs
        for p in self.pieces:
            p.filesystem = fs

    def __setstate__(self, state):
        """called when unpickling"""
        self.__dict__ = state
        if self.partition_names:
            # We do this because there is an error (bug?) when pickling
            # Arrow HivePartitioning objects
            part_dicts = {
                p: self.partitioning_dictionaries[i]
                for i, p in enumerate(self.partition_names)
            }
            self.partitioning = self.partitioning_cls(
                self.partitioning_schema, part_dicts
            )


class ParquetPiece:
    """Parquet dataset piece (file) information and Arrow objects to query
    metadata"""

    def __init__(self, frag, partitioning, partition_names):
        # We don't store the frag initially because we broadcast the dataset from rank 0,
        # and PyArrow has issues (un)pickling the frag, because it opens the file to access
        # metadata when pickling and/or unpickling. This can cause a massive slowdown
        # because a single process will try opening all the files in the dataset. Arrow 8
        # solved the issue when pickling, but I still see it when unpickling, reason
        # unknown (needs investigation). To check, simply pickle and unpickle the
        # bodo.io.parquet_io.ParquetDataset object on rank 0
        self._frag = None
        # needed to open the fragment (see frag property below)
        self.format = frag.format
        self.path = frag.path
        # number of rows in this piece after applying filters. This
        # is computed at runtime in `get_parquet_dataset`
        self._bodo_num_rows = 0
        self.partition_keys = []
        if partitioning is not None:
            # XXX these are not ordered by partitions or in inverse order for some reason
            self.partition_keys = ds._get_partition_keys(frag.partition_expression)
            self.partition_keys = [
                (
                    part_name,
                    partitioning.dictionaries[i]
                    .index(self.partition_keys[part_name])
                    .as_py(),
                )
                for i, part_name in enumerate(partition_names)
            ]

    @property
    def frag(self):
        """returns the Arrow ParquetFileFragment associated with this piece"""
        if self._frag is None:
            self._frag = self.format.make_fragment(
                self.path,
                self.filesystem,
            )
            del self.format
        return self._frag

    @property
    def metadata(self):
        """returns the Arrow metadata of this piece"""
        return self.frag.metadata

    @property
    def num_row_groups(self):
        """returns the number of row groups in this piece"""
        return self.frag.num_row_groups


@pt.overload
def get_fpath_without_protocol_prefix(
    fpath: str, protocol: str, parsed_url: ParseResult
) -> tuple[str, str]: ...


@pt.overload
def get_fpath_without_protocol_prefix(
    fpath: list[str], protocol: str, parsed_url: ParseResult
) -> tuple[list[str], str]: ...


def get_fpath_without_protocol_prefix(
    fpath: str | list[str], protocol: str, parsed_url: ParseResult
) -> tuple[str | list[str], str]:
    """
    Get the filepath(s) without the prefix associated with
    the protocol. e.g. in the s3 case, this will remove the
    "s3://" from the start of the path(s).

    Args:
        fpath (str | list[str]): Filepath or list of filepaths.
        protocol (str): Protocol being used for the paths.
            e.g. "" (local), "s3" (S3), etc.
        parsed_url (ParseResult): Properties such as scheme and netloc
            parsed from the filepath(s). This is typically the
            output of the 'parse_fpath' function.

    Returns:
        tuple[str | list[str], str]: Filepath(s) without the prefix
            and the prefix itself.
    """
    if protocol in {"abfs", "abfss"}:  # pragma: no cover
        # PyArrow AzureBlobFileSystem is initialized with account_name only
        # so the host / container name should be included in the files
        prefix = f"{protocol}://"

        def norm_path(p: str) -> str:
            url = urlparse(p)
            container = (
                parsed_url.netloc
                if parsed_url.username is None
                else parsed_url.username
            )
            return f"{container}{url.path}"

        if isinstance(fpath, list):
            mod_fpath = [norm_path(f) for f in fpath]
        else:
            mod_fpath = norm_path(fpath)
        return mod_fpath, prefix

    prefix = ""
    if protocol == "s3":
        prefix = "s3://"
    elif protocol == "s3a":
        prefix = "s3a://"
    elif protocol == "hdfs":
        # HDFS filesystem is initialized with host:port info. Once
        # initialized, the filesystem needs the <protocol>://<host><port>
        # prefix removed to query and access files
        prefix = f"{protocol}://{parsed_url.netloc}"
    elif protocol in {"gcs", "gs"}:
        prefix = f"{protocol}://"
    elif protocol == "hf":
        prefix = "hf://"

    if prefix:
        if isinstance(fpath, list):
            fpath_noprefix = [f[len(prefix) :] for f in fpath]
        else:
            fpath_noprefix = fpath[len(prefix) :]
    else:
        fpath_noprefix = fpath
    return fpath_noprefix, prefix


def fpath_without_protocol_prefix(fpath: str) -> str:
    """
    Get the filepath(s) without the prefix associated with
    the protocol. e.g. in the s3 case, this will remove the
    "s3://" from the start of the path(s).

    Args:
        fpath (str | list[str]): Filepath or list of filepaths.

    Returns:
        tuple[str | list[str], str]: Filepath(s) without the prefix
            and the prefix itself.
    """
    parsed_url = urlparse(fpath)
    protocol = parsed_url.scheme

    if protocol in {"abfs", "abfss", "wasb", "wasbs"}:
        # pragma: no cover
        # PyArrow AzureBlobFileSystem is initialized with account_name only
        # so the host / container name should be included in the files
        url = urlparse(fpath)
        container = (
            parsed_url.netloc if parsed_url.username is None else parsed_url.username
        )
        return f"{container}{url.path}"

    prefix = ""

    if protocol == "hdfs":
        # HDFS filesystem is initialized with host:port info. Once
        # initialized, the filesystem needs the <protocol>://<host><port>
        # prefix removed to query and access files
        prefix = f"{protocol}://{parsed_url.netloc}"
    elif protocol in {"s3", "s3a", "gcs", "gs"}:
        prefix = f"{protocol}://"

    return fpath[len(prefix) :]


def get_bodo_pq_dataset_from_fpath(
    fpath: str | list[str],
    protocol: str,
    parsed_url: ParseResult,
    fs: Any,
    partitioning: str | None,
    filters: pc.Expression | None = None,
    typing_pa_schema: pa.Schema | None = None,
) -> ParquetDataset | Exception:
    """
    Get a ParquetDataset object for a filepath (or a list of filepaths).
    If provided, the filters will be applied to prune the list
    of parquet files. Otherwise, all files are included in the dataset.
    This is used at both compile time and runtime.

    NOTE: This must be called on a single rank. The function will temporarily
    increase the number of IO threads PyArrow can use for parallelization.

    Args:
        fpath (str | list[str]): Filepath(s) for the dataset.
        protocol (str): Filesystem protocol.
        parsed_url (ParseResult): Details such as the scheme, netloc (i.e. the
            bucket in the S3 case), etc.
        fs (Any): Filesystem to use for reading data/metadata.
        partitioning (str | None): Partitioning scheme to use. Only 'hive'
            and None are supported.
        filters (pc.Expression, optional): Filters to apply to prune the set of
            files. Defaults to None.
        typing_pa_schema (Optional[pa.Schema], optional): Provide a schema
            to use. This is used at runtime to enforce the scheme
            inferred at compile-time. Defaults to None.

    Returns:
        ParquetDataset | Exception: Parquet dataset object with the relevant
            files. In case of an exception, the exception is returned so that
            the caller can handle error synchronization (since this function
            should be called from a single rank).
    """
    import bodo

    nthreads = 1  # Number of threads to use on this rank to collect metadata
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 1:
        nthreads = cpu_count // 2
    pa_default_io_thread_count = pa.io_thread_count()

    try:
        ev_pq_ds = tracing.Event("pq.ParquetDataset", is_parallel=False)
        if tracing.is_tracing():
            # only do the work of converting filters to string
            # if tracing is enabled
            ev_pq_ds.add_attribute("g_filters", str(filters))
        pa.set_io_thread_count(nthreads)

        fpath_noprefix, prefix = get_fpath_without_protocol_prefix(
            fpath, protocol, parsed_url
        )

        fpath_noprefix = expand_path_globs(fpath_noprefix, protocol, fs)

        dataset = pq.ParquetDataset(
            fpath_noprefix,
            filesystem=fs,
            # use_legacy_dataset=False is default by Arrow 15
            # Including arg introduces multiple warnings
            partitioning=partitioning,
            filters=filters,
        )

        num_files_before_filter = len(dataset.files)
        # If there are filters, files are filtered in ParquetDataset constructor
        dataset = ParquetDataset(dataset, prefix)

        # If typing schema is available, then use that as the baseline
        # schema to unify with, else get it from the dataset.
        # This is important for getting understandable errors in cases where
        # files have different schemas, some of which may or may not match
        # the iceberg schema. `get_dataset_schema` essentially gets the
        # schema of the first file. So, starting with that schema will
        # raise errors such as
        # "pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(TY)"
        # where TY is a column that originally had a different name.
        # Therefore, it's better to start with the expected schema,
        # and then raise the errors correctly after validation.
        if typing_pa_schema:
            # NOTE: typing_pa_schema must include partitions
            dataset.schema = typing_pa_schema

        if filters is not None:
            ev_pq_ds.add_attribute("num_pieces_before_filter", num_files_before_filter)
            ev_pq_ds.add_attribute("num_pieces_after_filter", len(dataset.pieces))
        ev_pq_ds.finalize()

        return dataset
    except Exception as e:
        # Import compiler lazily to access BodoError
        import bodo.decorators  # isort:skip # noqa

        # See note in pa_fs_list_dir_fnames
        # In some cases, OSError/FileNotFoundError can propagate
        # back to numba and come back as an InternalError.
        # where numba errors are hidden from the user.
        # See [BE-1188] for an example
        # Raising a bodo.utils.typing.BodoError lets messages come back and be seen by the user.
        if isinstance(e, IsADirectoryError):
            # We suppress Arrow's error message since it doesn't apply to Bodo
            # (the bit about doing a union of datasets)
            e = bodo.utils.typing.BodoError(LIST_OF_FILES_ERROR_MSG)
        elif isinstance(fpath, list) and isinstance(e, (OSError, FileNotFoundError)):
            e = bodo.utils.typing.BodoError(str(e) + LIST_OF_FILES_ERROR_MSG)
        else:
            e = bodo.utils.typing.BodoError(
                f"error from pyarrow: {type(e).__name__}: {str(e)}\n"
            )
        return e
    finally:
        # Restore pyarrow default IO thread count
        pa.set_io_thread_count(pa_default_io_thread_count)


# Create an mpi4py reduction function.
def pa_schema_unify_reduction(schema_a_and_row_count, schema_b_and_row_count, unused):
    # Attempt to unify the schemas, but if any schema is associated with a row
    # count of 0, disregard it.
    schema_a, count_a = schema_a_and_row_count
    schema_b, count_b = schema_b_and_row_count
    if count_a == 0 and count_b > 0:
        return (schema_b, count_b)
    if count_a > 0 and count_b == 0:
        return (schema_a, count_a)
    return (pa.unify_schemas([schema_a, schema_b]), count_a + count_b)


# Initialize local MPI operation for schema unification lazily
pa_schema_unify_mpi_op = None


def unify_schemas_across_ranks(dataset: ParquetDataset, total_rows_chunk: int):
    """
    Unify the dataset schema across all ranks.
    The dataset will be updated in place.

    Args:
        dataset (ParquetDataset): ParquetDataset whose schema
            should be unified and updated.
        total_rows_chunk (int): Number of rows in the row groups
            that the files allocated to this rank will read.

    Raises:
        bodo.utils.typing.BodoError: If schemas couldn't be unified.
    """
    import bodo

    global pa_schema_unify_mpi_op

    ev = tracing.Event("unify_schemas_across_ranks")
    error = None

    comm = MPI.COMM_WORLD
    if pa_schema_unify_mpi_op is None:
        pa_schema_unify_mpi_op = MPI.Op.Create(pa_schema_unify_reduction, commute=True)
    try:
        dataset.schema, _ = comm.allreduce(
            (dataset.schema, total_rows_chunk),
            pa_schema_unify_mpi_op,
        )
    except Exception as e:
        error = e

    # synchronize error state
    if comm.allreduce(error is not None, op=MPI.LOR):
        for error in comm.allgather(error):
            if error:
                # Import compiler lazily to access BodoError
                import bodo.decorators  # isort:skip # noqa

                msg = f"Schema in some files were different.\n{str(error)}"
                raise bodo.utils.typing.BodoError(msg)
    ev.finalize()


def unify_fragment_schema(dataset: ParquetDataset, piece: ParquetPiece, frag):
    """Unifies schema of *dataset* with incoming piece/fragment.

    Args:
        dataset (ParquetDataset): The Parquet dataset to update with the unified
            schema.
        piece (ParquetPiece): Piece corresponding to the fragment being unified.
        frag (pa.Dataset.Fragment): Fragment of the dataset to unify.

    Raises:
        bodo.utils.typing.BodoError: If the schemas cannot be unified
    """
    import bodo

    # Two files are compatible if arrow can unify their schemas.
    file_schema = frag.metadata.schema.to_arrow_schema()
    fileset_schema_names = set(file_schema.names)
    # Check the names are the same because pa.unify_schemas
    # will unify a schema where a column is in 1 file but not
    # another.
    dataset_schema_names = set(dataset.schema.names) - set(dataset.partition_names)
    # File schema can only be a (potentially) more restrictive
    # version of the starting schema, therefore, the file shouldn't
    # have extra columns. Any columns that are expected but are
    # missing from the file will be filled with nulls at read time.
    added_columns = fileset_schema_names - dataset_schema_names
    if added_columns:
        # Import compiler lazily to access BodoError
        import bodo.decorators  # isort:skip # noqa

        msg = f"Schema in {piece} was different. File contains column(s) {added_columns} not expected in the dataset.\n"
        raise bodo.utils.typing.BodoError(msg)
    try:
        dataset.schema = unify_schemas([dataset.schema, file_schema], "permissive")
    except Exception as e:
        # Import compiler lazily to access BodoError
        import bodo.decorators  # isort:skip # noqa

        msg = f"Schema in {piece} was different.\n{str(e)}"
        raise bodo.utils.typing.BodoError(msg)


def populate_row_counts_in_pq_dataset_pieces(
    dataset: ParquetDataset,
    fpath: str | list[str],
    protocol: str,
    validate_schema: bool,
    filters: pc.Expression | None = None,
):
    """
    Populate the row counts for each piece in a ParquetDataset
    by applying the filters. This uses the PyArrow dataset
    APIs to look at the metadata to do row-group level
    filtering and get the exact row counts. This may lead
    to some data read, however, Arrow tries to limit it to a
    minimum wherever possible.

    NOTE: This function must be called on all ranks.

    Args:
        dataset (ParquetDataset): ParquetDataset object with details
            about the schema, files, etc.
        fpath (str | list[str]): Original filepath(s) used to create
            the dataset. Note that these are only used in some warning
            messages and not used for actual file discovery.
            XXX TODO Explore getting rid of this dependency and using
            the file paths from the dataset directly.
        protocol (str): Filesystem protocol. Used for warning messages
            when using remote filesystems.
        validate_schema (bool): Whether the schema should be validated
            for each file in the dataset. If set to true, it will
            also update the dataset schema to be the unification of
            the original schema and the schemas of the files. Note that
            this will only validate schema for the set of files its
            processing.
        filters (pc.Expression, optional): Arrow expression filters
            to apply. Defaults to None.
    """
    import bodo

    ev_row_counts = tracing.Event("get_row_counts")
    # getting row counts and validating schema requires reading
    # the file metadata from the parquet files and is very expensive
    # for datasets consisting of many files, so we do this in parallel
    if tracing.is_tracing():
        ev_row_counts.add_attribute("g_num_pieces", len(dataset.pieces))
        ev_row_counts.add_attribute("g_filters", str(filters))
    ds_scan_time = 0.0
    num_pieces = len(dataset.pieces)
    start = bodo.get_start(num_pieces, bodo.get_size(), bodo.get_rank())
    end = bodo.get_end(num_pieces, bodo.get_size(), bodo.get_rank())
    total_rows_chunk = 0
    total_row_groups_chunk = 0
    total_row_groups_size_chunk = 0

    if filters is not None:
        my_random = random.Random(37)
        pieces = my_random.sample(dataset.pieces, k=len(dataset.pieces))
    else:
        pieces = dataset.pieces

    fpaths = [p.path for p in pieces[start:end]]
    # Presumably the work is partitioned more or less equally among ranks,
    # and we are mostly (or just) reading metadata, so we assign four IO
    # threads to every rank
    nthreads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), 4)
    pa_default_io_thread_count = pa.io_thread_count()
    pa.set_io_thread_count(nthreads)
    pa.set_cpu_count(nthreads)
    # Use dataset scanner API to get exact row counts when
    # filter is applied. Arrow will try to calculate this by
    # by reading only the file's metadata, and if it needs to
    # access data it will read as less as possible (only the
    # required columns and only subset of row groups if possible)
    error = None
    try:
        dataset_ = ds.dataset(
            fpaths,
            filesystem=dataset.filesystem,
            partitioning=dataset.partitioning,
        )

        for piece, frag in zip(pieces[start:end], dataset_.get_fragments()):
            # The validation (and unification) step needs to happen before the
            # scan on the fragment since otherwise it will fail in case the
            # file schema doesn't match the dataset schema exactly.
            # Currently this is only applicable for Iceberg reads.
            if validate_schema:
                unify_fragment_schema(dataset, piece, frag)

            t0 = time.time()
            # We use the expected schema instead of the file schema. This schema
            # should be a less-restrictive superset of the file schema (after the
            # unification step above), so it should be valid.
            row_count = frag.scanner(
                schema=dataset.schema,
                filter=filters,
                use_threads=True,
            ).count_rows()
            ds_scan_time += time.time() - t0
            piece._bodo_num_rows = row_count
            total_rows_chunk += row_count
            total_row_groups_chunk += frag.num_row_groups
            total_row_groups_size_chunk += sum(
                rg.total_byte_size for rg in frag.row_groups
            )

    except Exception as e:
        error = e
    finally:
        # Restore pyarrow default IO thread count
        pa.set_io_thread_count(pa_default_io_thread_count)

    # synchronize error state
    comm = MPI.COMM_WORLD
    if comm.allreduce(error is not None, op=MPI.LOR):
        for error in comm.allgather(error):
            if error:
                if isinstance(fpath, list) and isinstance(
                    error, (OSError, FileNotFoundError)
                ):
                    # Import compiler lazily to access BodoError
                    import bodo.decorators  # isort:skip # noqa

                    raise bodo.utils.typing.BodoError(
                        str(error) + LIST_OF_FILES_ERROR_MSG
                    )
                raise error

    # Now unify the schemas across all ranks.
    if validate_schema:
        unify_schemas_across_ranks(dataset, total_rows_chunk)

    dataset._bodo_total_rows = comm.allreduce(total_rows_chunk, op=MPI.SUM)
    total_num_row_groups = comm.allreduce(total_row_groups_chunk, op=MPI.SUM)
    total_row_groups_size = comm.allreduce(total_row_groups_size_chunk, op=MPI.SUM)
    pieces_rows = np.array([p._bodo_num_rows for p in dataset.pieces])
    # communicate row counts to everyone
    pieces_rows = comm.allreduce(pieces_rows, op=MPI.SUM)
    for p, nrows in zip(dataset.pieces, pieces_rows):
        p._bodo_num_rows = nrows
    if (
        bodo.get_rank() == 0
        and total_num_row_groups < bodo.get_size()
        and total_num_row_groups != 0
    ):
        if isinstance(fpath, list) and len(fpath) > 5:
            fpath_tidbit = "[{}, ... {} more files]".format(
                ", ".join(fpath[:5]), len(fpath) - 5
            )
        else:
            fpath_tidbit = fpath

        warnings.warn(
            bodo.BodoWarning(
                f"Total number of row groups in parquet dataset {fpath_tidbit} ({total_num_row_groups}) is too small for effective IO parallelization."
                f"For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()}). For more details, refer to https://docs.bodo.ai/latest/file_io/#parquet-section."
            )
        )

    # print a warning if average row group size < 1 MB and reading from remote filesystem
    if total_num_row_groups == 0:
        avg_row_group_size_bytes = 0
    else:
        avg_row_group_size_bytes = total_row_groups_size // total_num_row_groups
    if (
        bodo.get_rank() == 0
        and total_row_groups_size >= 20 * 1048576
        and avg_row_group_size_bytes < 1048576
        and protocol in REMOTE_FILESYSTEMS
    ):
        warnings.warn(
            bodo.BodoWarning(
                f"Parquet average row group size is small ({avg_row_group_size_bytes} bytes) and can have negative impact on performance when reading from remote sources"
            )
        )
    if tracing.is_tracing():
        ev_row_counts.add_attribute("g_total_num_row_groups", total_num_row_groups)
        ev_row_counts.add_attribute("total_scan_time", ds_scan_time)
        # get 5-number summary for rowcounts:
        # (min, max, 25, 50 -median-, 75 percentiles)
        data = np.array([p._bodo_num_rows for p in dataset.pieces])
        quartiles = np.percentile(data, [25, 50, 75])
        ev_row_counts.add_attribute("g_row_counts_min", data.min())
        ev_row_counts.add_attribute("g_row_counts_Q1", quartiles[0])
        ev_row_counts.add_attribute("g_row_counts_median", quartiles[1])
        ev_row_counts.add_attribute("g_row_counts_Q3", quartiles[2])
        ev_row_counts.add_attribute("g_row_counts_max", data.max())
        ev_row_counts.add_attribute("g_row_counts_mean", data.mean())
        ev_row_counts.add_attribute("g_row_counts_std", data.std())
        ev_row_counts.add_attribute("g_row_counts_sum", data.sum())
    ev_row_counts.finalize()


def get_parquet_dataset(
    fpath,
    get_row_counts: bool = True,
    filters: pc.Expression | None = None,
    storage_options: dict | None = None,
    read_categories: bool = False,
    tot_rows_to_read: int | None = None,
    typing_pa_schema: pa.Schema | None = None,
    partitioning: str | None = "hive",
) -> ParquetDataset:
    """
    Get ParquetDataset object for 'fpath' and set the number of total rows as an
    attribute. Also, sets the number of rows per file as an attribute of
    ParquetDatasetPiece objects.

    Args:
        filters: Used for predicate pushdown which prunes the unnecessary pieces.
        read_categories: Read categories of DictionaryArray and store in returned dataset
            object, used during typing.
        get_row_counts: This is only true at runtime, and indicates that we need
            to get the number of rows of each piece in the parquet dataset.
        tot_rows_to_read: total number of rows to read from dataset. Used at runtime
            for example if doing df.head(tot_rows_to_read) where df is the output of
            read_parquet()
        typing_pa_schema: PyArrow schema determined at compile time. When provided,
            we should validate that the unified schema of all files matches this schema,
            and throw an error otherwise. Currently this is only used in runtime.
            https://bodo.atlassian.net/browse/BE-2787
    """

    # NOTE: This function obtains the metadata for a parquet dataset and works
    # in the same way regardless of whether the read is going to be parallel or
    # replicated. In all cases rank 0 will get the ParquetDataset from pyarrow,
    # broadcast it to all ranks, and they will divide the work of getting the
    # number of rows in each file of the dataset

    # get_parquet_dataset can be called both at both compile and run time. We
    # only want to trace it at run time, so take advantage of get_row_counts flag
    # to know if this is runtime
    if get_row_counts:
        ev = tracing.Event("get_parquet_dataset")

    # We add dummy parameter in _gen_pq_reader_py for Numba typing
    # but it can get in the way of other Filesystem builders
    if storage_options:
        storage_options.pop("bodo_dummy", None)

    comm = MPI.COMM_WORLD
    fpath, parsed_url, protocol = parse_fpath(fpath)

    # Getting row counts and schema validation is going to be
    # distributed across ranks, so every rank will need a filesystem
    # object to query the metadata of their assigned pieces.
    # We have seen issues in the past with broadcasting some filesystem
    # objects (e.g. s3) and broadcasting the filesystem adds extra time,
    # so instead we initialize the filesystem before the broadcast.
    # That way all ranks do it in parallel at the same time.
    fs = getfs(fpath, protocol, storage_options, get_row_counts)
    validate_schema = bodo.parquet_validate_schema if get_row_counts else False

    dataset_or_err: ParquetDataset | Exception | None = None
    if bodo.get_rank() == 0:
        dataset_or_err = get_bodo_pq_dataset_from_fpath(
            fpath,
            protocol,
            parsed_url,
            fs,
            partitioning,
            filters,
            typing_pa_schema,
        )
    if get_row_counts:
        ev_bcast = tracing.Event("bcast dataset")
    dataset_or_err = comm.bcast(dataset_or_err)
    if get_row_counts:
        ev_bcast.finalize()

    if isinstance(dataset_or_err, Exception):  # pragma: no cover
        error = dataset_or_err
        raise error
    dataset = pt.cast(ParquetDataset, dataset_or_err)

    # As mentioned above, we don't want to broadcast the filesystem because it
    # adds time (so initially we didn't include it in the dataset). We add
    # it to the dataset now that it's been broadcasted
    dataset.set_fs(fs)

    if get_row_counts and tot_rows_to_read == 0:
        get_row_counts = validate_schema = False

    if get_row_counts:
        populate_row_counts_in_pq_dataset_pieces(
            dataset,
            fpath,
            protocol,
            validate_schema,
            filters,
        )

    if read_categories:
        _add_categories_to_pq_dataset(dataset)

    if get_row_counts:
        ev.finalize()

    return dataset


def filter_row_groups_from_start_of_dataset_heuristic(
    len_fpaths: int, start_offset: int, expr_filter: pc.Expression | None
) -> bool:
    """
    Heuristic to determine whether or not to prune individual
    row groups from the start of a file.

    Args:
        len_fpaths (int): Total number of files being read.
        start_offset (int): Starting row offset into the first fiel.
        expr_filter (pc.Expression | None): Filter to apply.

    Returns:
        bool: Whether or not to prune row groups.
    """
    # ------- row group filtering -------
    # Ranks typically will not read all the row groups from their list of
    # files (they will skip some rows at the beginning of the first file and
    # some rows at the end of the last one).
    # To make sure this rank only reads from the minimum necessary row groups,
    # we can create a new dataset object composed of row group fragments
    # instead of file fragments. We need to do it like this because Arrow's
    # scanner doesn't support skipping rows.
    # For this approach, we need to get row group metadata which can be very
    # expensive when reading from remote filesystems. Also, row group filtering
    # typically only benefits when the rank reads from a small set of files
    # (since the filtering only applies to the first and last file).
    # So we only filter based on this heuristic:
    # Filter row groups if the list of files is very small, or if it is <= 10
    # and this rank needs to skip rows of the first file.
    # TODO see if getting row counts with filter pushdown could be worthwhile
    # in some specific cases, and integrate that into this heuristic.
    return (expr_filter is None) and (
        len_fpaths <= 3 or (start_offset > 0 and len_fpaths <= 10)
    )


def filter_row_groups_from_start_of_dataset(
    dataset: ds.FileSystemDataset,
    start_offset: int,
    max_rows_to_read: int,
    pq_format: ds.ParquetFileFormat,
) -> tuple[ds.FileSystemDataset, int]:
    """
    Filter row groups from the start of the dataset.
    See rationale in the description of
    filter_row_groups_from_start_of_dataset_heuristic.

    Args:
        dataset (ds.FileSystemDataset): Original dataset.
        start_offset (int): Starting row offset into the
            first file.
        max_rows_to_read (int): Maximum number of rows
            this process will read from the files.
        pq_format (ds.ParquetFileFormat): FileFormat
            used for constructing the new dataset.

    Returns:
        tuple[ds.FileSystemDataset, int]:
            - New dataset.
            - New offest into the first piece/row-group.
    """
    new_frags = []
    # Total number of rows of all the row groups we iterate through
    count_rows = 0
    # Track total rows that this rank will read from row groups we iterate
    # through
    rows_added = 0
    start_row_first_rg = -1
    for frag in dataset.get_fragments():
        # Each fragment is a parquet file.
        # For reference, this is basically the same logic as in
        # ArrowReader::init_arrow_reader() and just adapted from there.
        # Get the file's row groups that this rank will read from
        row_group_ids = []
        for rg in frag.row_groups:
            num_rows_rg = rg.num_rows
            if start_offset < count_rows + num_rows_rg:
                # Rank needs to read from this row group
                if rows_added == 0:
                    # This is the first row group the rank will read from
                    start_row_first_rg = start_offset - count_rows
                    rows_added_from_rg = min(
                        num_rows_rg - start_row_first_rg, max_rows_to_read
                    )
                else:
                    rows_added_from_rg = min(num_rows_rg, max_rows_to_read - rows_added)
                rows_added += rows_added_from_rg
                row_group_ids.append(rg.id)
            count_rows += num_rows_rg
            if rows_added >= max_rows_to_read:
                break
        # XXX frag.subset(row_group_ids) is expensive on remote filesystems
        # with datasets composed of many files and row groups
        new_frags.append(frag.subset(row_group_ids=row_group_ids))
        if rows_added >= max_rows_to_read:
            break

    # New dataset:
    new_dataset = ds.FileSystemDataset(
        new_frags, dataset.schema, pq_format, filesystem=dataset.filesystem
    )

    assert start_row_first_rg >= 0

    return new_dataset, start_row_first_rg


def schema_with_dict_cols(schema: pa.Schema, str_as_dict_cols: list[str]) -> pa.Schema:
    """
    Helper function to get a PyArrow schema object
    from an existing Schema by replacing certain
    columns to be dictionary-encoded.

    Args:
        schema (pa.Schema): Original schema.
        str_as_dict_cols (list[str]): Column to dict-encode.

    Returns:
        pa.Schema: New schema.
    """
    if len(str_as_dict_cols) == 0:
        return schema
    new_fields: list[pa.Field] = []
    dict_col_set = set(str_as_dict_cols)
    for i, name in enumerate(schema.names):
        if name in dict_col_set:
            old_field = schema.field(i)
            new_field = old_field.with_type(pa.dictionary(pa.int32(), old_field.type))
            new_fields.append(new_field)
        else:
            new_fields.append(schema.field(i))
    return pa.schema(new_fields)


def get_scanner_batches(
    fpaths,
    filters: pc.Expression | None,
    selected_fields: list[int],
    avg_num_pieces: float,
    is_parallel: bool,
    filesystem,
    str_as_dict_cols,
    start_offset: int,  # starting row offset in the pieces this process is going to read
    rows_to_read: int,  # total number of rows this process is going to read
    partitioning,
    schema: pa.Schema,
    batch_size: int,
    batch_readahead: int,
    fragment_readahed: int,
):
    """return RecordBatchReader for dataset of 'fpaths' that contain the rows
    that match filters (or all rows if filters is None). Only project the
    fields in selected_fields"""
    import pyarrow as pa

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_threads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), cpu_count)
    max_threads = min(int(os.environ.get("BODO_MAX_IO_THREADS", 16)), cpu_count)
    # TODO Unset this after the read??
    if (
        is_parallel
        and len(fpaths) > max_threads
        and len(fpaths) / avg_num_pieces >= 2.0
    ):
        # assign more threads to ranks that have to read
        # many more files than others
        pa.set_io_thread_count(max_threads)
        pa.set_cpu_count(max_threads)
    else:
        pa.set_io_thread_count(default_threads)
        pa.set_cpu_count(default_threads)

    pq_format = ds.ParquetFileFormat(dictionary_columns=str_as_dict_cols)

    # Set columns to be read as dictionary encoded in schema
    schema = schema_with_dict_cols(schema, str_as_dict_cols)

    dataset = ds.dataset(
        fpaths,
        filesystem=filesystem,
        partitioning=partitioning,
        schema=schema,
        format=pq_format,
    )
    col_names = dataset.schema.names
    selected_names = [col_names[field_num] for field_num in selected_fields]

    if filter_row_groups_from_start_of_dataset_heuristic(
        len(fpaths), start_offset, filters
    ):
        # The starting offset the Parquet reader knows about is from the first
        # file, not the first row group, so we need to communicate this back to C++
        dataset, start_offset = filter_row_groups_from_start_of_dataset(
            dataset, start_offset, rows_to_read, pq_format
        )

    rb_reader = dataset.scanner(
        # XXX Specifying "__filename" as one of the columns
        # will create a column with the filename. We might
        # want to replace our custom filename handling with
        # this at some point.
        columns=selected_names,
        filter=filters,
        batch_size=batch_size,
        use_threads=True,
        batch_readahead=batch_readahead,
        fragment_readahead=fragment_readahed,
        # XXX Specify memory pool?
    ).to_reader()
    return rb_reader, start_offset


# XXX Move this to ParquetDataset class?
def _add_categories_to_pq_dataset(pq_dataset):
    """adds categorical values for each categorical column to the Parquet dataset
    as '_category_info' attribute
    """
    import pyarrow as pa
    from mpi4py import MPI

    import bodo

    # NOTE: shouldn't be possible
    if len(pq_dataset.pieces) < 1:  # pragma: no cover
        # Import compiler lazily to access BodoError
        import bodo.decorators  # isort:skip # noqa

        raise bodo.utils.typing.BodoError(
            "No pieces found in Parquet dataset. Cannot get read categorical values"
        )

    pa_schema = pq_dataset.schema
    cat_col_names = [
        c
        for c in pa_schema.names
        if isinstance(pa_schema.field(c).type, pa.DictionaryType)
        and c not in pq_dataset.partition_names
    ]

    # avoid more work if no categorical columns
    if len(cat_col_names) == 0:
        pq_dataset._category_info = {}
        return

    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        try:
            # read categorical values from first row group of first file
            table_sample = pq_dataset.pieces[0].frag.head(100, columns=cat_col_names)
            # NOTE: assuming DictionaryArray has only one chunk
            category_info = {
                c: tuple(table_sample.column(c).chunk(0).dictionary.to_pylist())
                for c in cat_col_names
            }
            del table_sample  # release I/O resources ASAP
        except Exception as e:
            comm.bcast(e)
            raise e

        comm.bcast(category_info)
    else:
        category_info = comm.bcast(None)
        if isinstance(category_info, Exception):  # pragma: no cover
            error = category_info
            raise error

    pq_dataset._category_info = category_info


def get_pandas_metadata(schema) -> tuple[list[str | dict], dict[str, bool | None]]:
    # find pandas index column if any
    # TODO: other pandas metadata like dtypes needed?
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html
    index_cols = []
    # column_name -> is_nullable (or None if unknown)
    nullable_from_metadata: defaultdict[str, bool | None] = defaultdict(lambda: None)
    KEY = b"pandas"
    if schema.metadata is not None and KEY in schema.metadata:
        pandas_metadata = json.loads(schema.metadata[KEY].decode("utf8"))
        if pandas_metadata is None:
            return [], nullable_from_metadata

        index_cols = pandas_metadata["index_columns"]
        # ignore non-str/dict index metadata
        index_cols = [
            index_col
            for index_col in index_cols
            if isinstance(index_col, str) or isinstance(index_col, dict)
        ]

        for col_dict in pandas_metadata["columns"]:
            col_name = col_dict["name"]
            col_pd_type = col_dict["pandas_type"]
            if (
                col_pd_type.startswith("int") or col_pd_type.startswith("float")
            ) and col_name is not None:
                col_np_type = col_dict["numpy_type"]
                if col_np_type.startswith("Int") or col_np_type.startswith("Float"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False
    return index_cols, nullable_from_metadata


def get_str_columns_from_pa_schema(pa_schema: pa.Schema) -> list[str]:
    """
    Get the list of string type columns in the schema.
    """
    str_columns: list[str] = []
    for col_name in pa_schema.names:
        field = pa_schema.field(col_name)
        if field.type in (pa.string(), pa.large_string()):
            str_columns.append(col_name)
    return str_columns


def _pa_schemas_match(pa_schema1: pa.Schema, pa_schema2: pa.Schema) -> bool:
    """check if Arrow schemas match or not"""
    # check column names
    if pa_schema1.names != pa_schema2.names:
        return False

    # check type matches
    try:
        unify_schemas([pa_schema1, pa_schema2])
    except Exception:
        return False

    return True


def _get_sample_pq_pieces(
    pq_dataset: ParquetDataset | None,
):
    """get a sample of pieces in the Parquet dataset to avoid the overhead of opening
    every file in compile time.

    Args:
        pq_dataset: input Parquet dataset

    Returns:
        list(ParquetPiece): A sample of filtered pieces
    """
    # Bodo IO uses None to represent an empty dataset, which has 0 pieces
    pieces = pq_dataset.pieces if pq_dataset else []

    # a sample of N files where N is the number of ranks. Each rank looks at
    # the metadata of a different random file
    if len(pieces) > bodo.get_size():
        import random

        my_random = random.Random(37)
        pieces = my_random.sample(pieces, bodo.get_size())
    else:
        pieces = pieces

    return pieces


def determine_str_as_dict_columns(pq_dataset, pa_schema, str_columns: list) -> set:
    """
    Determine which string columns (str_columns) should be read by Arrow as
    dictionary encoded arrays, based on this heuristic:
      calculating the ratio of total_uncompressed_size of the column vs number
      of values.
      If the ratio is less than READ_STR_AS_DICT_THRESHOLD we read as DICT.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    if len(str_columns) == 0:
        return set()  # no string as dict columns

    # Get a sample of Parquet pieces to avoid opening every file in compile time
    pieces = _get_sample_pq_pieces(pq_dataset)

    # Sort the list to ensure same order on all ranks. This is
    # important for correctness.
    str_columns = sorted(str_columns)
    total_uncompressed_sizes = np.zeros(len(str_columns), dtype=np.int64)
    total_uncompressed_sizes_recv = np.zeros(len(str_columns), dtype=np.int64)
    if bodo.get_rank() < len(pieces):
        piece = pieces[bodo.get_rank()]
        try:
            metadata = piece.metadata
            for i in range(piece.num_row_groups):
                for j, col_name in enumerate(str_columns):
                    idx = pa_schema.get_field_index(col_name)
                    total_uncompressed_sizes[j] += (
                        metadata.row_group(i).column(idx).total_uncompressed_size
                    )
            num_rows = metadata.num_rows
        except Exception as e:
            if isinstance(e, (OSError, FileNotFoundError)):
                # skip the path that produced the error (error will be reported at runtime)
                num_rows = 0
            else:
                raise
    else:
        num_rows = 0
    total_rows = comm.allreduce(num_rows, op=MPI.SUM)
    if total_rows == 0:
        return set()  # no string as dict columns
    comm.Allreduce(total_uncompressed_sizes, total_uncompressed_sizes_recv, op=MPI.SUM)
    str_column_metrics = total_uncompressed_sizes_recv / total_rows
    str_as_dict = set()
    for i, metric in enumerate(str_column_metrics):
        if metric < READ_STR_AS_DICT_THRESHOLD:
            col_name = str_columns[i]
            str_as_dict.add(col_name)
    return str_as_dict


def parquet_file_schema(
    file_name,
    selected_columns,
    storage_options=None,
    input_file_name_col=None,
    read_as_dict_cols=None,
    use_hive: bool = True,
) -> FileSchema:
    """get parquet schema from file using Parquet dataset and Arrow APIs"""
    # Import compiler lazily to access BodoError
    import bodo
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.helpers import _get_numba_typ_from_pa_typ
    from bodo.libs.dict_arr_ext import dict_str_arr_type

    col_names = []
    col_types = []
    # during compilation we only need the schema and it has to be the same for
    # all processes, so we can set parallel=True to just have rank 0 read
    # the dataset information and broadcast to others
    pq_dataset = get_parquet_dataset(
        file_name,
        get_row_counts=False,
        storage_options=storage_options,
        read_categories=True,
        partitioning="hive" if use_hive else None,
    )
    pq_dataset = parquet_dataset_unify_nulls(pq_dataset)

    partition_names = pq_dataset.partition_names
    pa_schema = pq_dataset.schema

    # Get list of string columns
    str_columns = get_str_columns_from_pa_schema(pa_schema)
    # Convert to set (easier for set operations like intersect and union)
    str_columns_set = set(str_columns)
    if read_as_dict_cols is None:
        read_as_dict_cols = []
    read_as_dict_cols = set(read_as_dict_cols)
    # If user-provided list has any columns that are not string
    # type, show a warning.
    non_str_columns_in_read_as_dict_cols = read_as_dict_cols - str_columns_set
    if len(non_str_columns_in_read_as_dict_cols) > 0:
        if bodo.get_rank() == 0:
            warnings.warn(
                f"The following columns are not of datatype string and hence cannot be read with dictionary encoding: {non_str_columns_in_read_as_dict_cols}",
                bodo.BodoWarning,
            )
    # Remove non-string columns from read_as_dict_cols
    read_as_dict_cols.intersection_update(str_columns_set)
    # Remove read_as_dict_cols from str_columns (no need to compute heuristic on these)
    str_columns_set = str_columns_set - read_as_dict_cols
    # Match the list with the set. We've only removed entries, so a filter is sufficient.
    # Order of columns in the list is important between different ranks,
    # so either we do this, or sort.
    str_columns = [x for x in str_columns if x in str_columns_set]
    # Get the set of columns to be read with dictionary encoding based on heuristic
    str_as_dict = determine_str_as_dict_columns(pq_dataset, pa_schema, str_columns)
    # Add user-provided columns to the list
    str_as_dict.update(read_as_dict_cols)

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with
    # names of list types columns (arrow 0.17.0)
    # col_names is an array that contains all the column's name and
    # index's name if there is one, otherwise "__index__level_0__"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names
    index_cols, nullable_from_metadata = get_pandas_metadata(pa_schema)
    index_col_names: set[str] = {name for name in index_cols if isinstance(name, str)}
    col_types_total = []
    is_supported_list = []
    arrow_types = []
    for i, c in enumerate(col_names):
        if c in partition_names:
            continue
        field = pa_schema.field(c)
        dtype, supported = _get_numba_typ_from_pa_typ(
            field,
            c in index_col_names,
            nullable_from_metadata[c],
            pq_dataset._category_info,
            str_as_dict=c in str_as_dict,
        )
        col_types_total.append(dtype)
        is_supported_list.append(supported)
        # Store the unsupported arrow type for future
        # error messages.
        arrow_types.append(field.type)

    # add partition column data types if any
    if partition_names:
        col_types_total += [
            _get_partition_cat_dtype(pq_dataset.partitioning_dictionaries[i])
            for i in range(len(partition_names))
        ]
        # All partition column types are supported by default.
        is_supported_list.extend([True] * len(partition_names))
        # Extend arrow_types for consistency. Here we use None
        # because none of these are actually in the pq file.
        arrow_types.extend([None] * len(partition_names))

    # add input_file_name column data type if one is specified
    if input_file_name_col is not None:
        col_names += [input_file_name_col]
        col_types_total += [dict_str_arr_type]
        # input_file_name column is a dictionary-encoded string array which is supported by default.
        is_supported_list.append(True)
        # Extend arrow_types for consistency. Here we use None
        # because this column isn't actually in the pq file.
        arrow_types.append(None)

    # Map column names to index to allow efficient search
    col_names_map = {c: i for i, c in enumerate(col_names)}

    # if no selected columns, set it to all of them.
    if selected_columns is None:
        selected_columns = col_names

    # make sure selected columns are in the schema
    for c in selected_columns:
        if c not in col_names_map:
            # Import compiler lazily to access BodoError
            import bodo.decorators  # isort:skip # noqa

            raise bodo.utils.typing.BodoError(
                f"Selected column {c} not in Parquet file schema"
            )
    for index_col in index_cols:
        if not isinstance(index_col, dict) and index_col not in selected_columns:
            # if index_col is "__index__level_0__" or some other name, append it.
            # If the index column is not selected when reading parquet, the index
            # should still be included.
            selected_columns.append(index_col)

    # Convert dictionary columns to use int32 indices
    # since our c++ dict string array uses int32 indices
    for i in range(len(arrow_types)):
        if isinstance(arrow_types[i], pa.DictionaryType):
            arrow_types[i] = pa.dictionary(pa.int32(), arrow_types[i].value_type)
            pa_schema = pa_schema.set(i, pa_schema.field(i).with_type(arrow_types[i]))

    col_names = selected_columns
    col_indices = []
    col_types = []
    unsupported_columns = []
    unsupported_arrow_types = []
    for i, c in enumerate(col_names):
        col_idx = col_names_map[c]
        col_indices.append(col_idx)
        col_types.append(col_types_total[col_idx])
        if not is_supported_list[col_idx]:
            unsupported_columns.append(i)
            unsupported_arrow_types.append(arrow_types[col_idx])

    # TODO: close file?
    return (
        col_names,
        col_types,
        index_cols,
        col_indices,
        partition_names,
        unsupported_columns,
        unsupported_arrow_types,
        pa_schema,
    )


def _get_partition_cat_dtype(dictionary):
    """get categorical dtype for Parquet partition set"""
    from numba.core import types

    # Import compiler lazily
    import bodo
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.pd_categorical_ext import (
        CategoricalArrayType,
        PDCategoricalDtype,
    )

    # using 'dictionary' instead of 'keys' attribute since 'keys' may not have the
    # right data type (e.g. string instead of int64)
    assert dictionary is not None
    S = dictionary.to_pandas()
    elem_type = bodo.typeof(S).dtype
    if isinstance(elem_type, (types.Integer)):
        cat_dtype = PDCategoricalDtype(tuple(S), elem_type, False, int_type=elem_type)
    else:
        cat_dtype = PDCategoricalDtype(tuple(S), elem_type, False)
    return CategoricalArrayType(cat_dtype)


def parquet_dataset_unify_nulls(
    dataset: ParquetDataset,
) -> ParquetDataset:
    """
    Gets the ParquetDataset from fpath, unifying types of null columns if present.

    NOTE: This function is intended to handle
    the common case where the first file opened contains some null columns which have
    non-null values in other files.
    """
    # If there are no null columns, skip unify step.
    if not any(pa.types.is_null(typ) for typ in dataset.schema.types):
        return dataset

    # Open the dataset similar to
    # https://github.com/bodo-ai/Bodo/blob/294d0ea13ebba84f07d8e6ebfe297449c1e0b77b/bodo/io/parquet_pio.py#L717
    pieces = dataset.pieces
    fpaths = [p.path for p in dataset.pieces]
    dataset_ = ds.dataset(
        fpaths,
        filesystem=dataset.filesystem,
        partitioning=dataset.partitioning,
    )

    # If there are nulls in the schema, inspect the fragments
    # until the null columns can be resolved to a non-null type.
    row_count = 0
    for piece, frag in zip(pieces, dataset_.get_fragments()):
        unify_fragment_schema(dataset, piece, frag)
        row_count += piece._bodo_num_rows
        if not any(pa.types.is_null(typ) for typ in dataset.schema.types):
            break

    comm = MPI.COMM_WORLD

    if comm.Get_size() > 1:
        unify_schemas_across_ranks(dataset, row_count)

    return dataset
