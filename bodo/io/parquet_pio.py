# Copyright (C) 2022 Bodo Inc. All rights reserved.
import os
import warnings
from collections import defaultdict
from glob import has_magic
from typing import Optional
from urllib.parse import urlparse

import llvmlite.binding as ll
import numba
import numpy as np
import pyarrow  # noqa
import pyarrow as pa  # noqa
import pyarrow.dataset as ds
from numba.core import types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    models,
    overload,
    register_model,
    unbox,
)
from pyarrow._fs import PyFileSystem
from pyarrow.fs import FSSpecHandler

import bodo
import bodo.utils.tracing as tracing
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.io.fs_io import get_hdfs_fs, get_s3_fs_from_path
from bodo.io.helpers import _get_numba_typ_from_pa_typ
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.distributed_api import get_end, get_start
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    FileInfo,
    FileSchema,
    get_overload_const_str,
)

REMOTE_FILESYSTEMS = {"s3", "gcs", "gs", "http", "hdfs", "abfs", "abfss"}
# the ratio of total_uncompressed_size of a Parquet string column vs number of values,
# below which we read as dictionary-encoded string array
READ_STR_AS_DICT_THRESHOLD = 1.0

list_of_files_error_msg = ". Make sure the list/glob passed to read_parquet() only contains paths to files (no directories)"


class ParquetPredicateType(types.Type):
    """Type for predicate list for Parquet filtering (e.g. [["a", "==", 2]]).
    It is just a Python object passed as pointer to C++
    """

    def __init__(self):
        super(ParquetPredicateType, self).__init__(name="ParquetPredicateType()")


parquet_predicate_type = ParquetPredicateType()
types.parquet_predicate_type = parquet_predicate_type  # type: ignore
register_model(ParquetPredicateType)(models.OpaqueModel)


@unbox(ParquetPredicateType)
def unbox_parquet_predicate_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(ParquetPredicateType)
def box_parquet_predicate_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


class ParquetFileInfo(FileInfo):
    """FileInfo object passed to ForceLiteralArg for
    file name arguments that refer to a parquet dataset"""

    def __init__(
        self,
        columns,
        storage_options=None,
        input_file_name_col=None,
        read_as_dict_cols=None,
        use_hive=True,
    ):
        self.columns = columns  # columns to select from parquet dataset
        self.storage_options = storage_options
        self.input_file_name_col = input_file_name_col
        self.read_as_dict_cols = read_as_dict_cols
        self.use_hive = use_hive
        super().__init__()

    def _get_schema(self, fname):
        try:
            return parquet_file_schema(
                fname,
                selected_columns=self.columns,
                storage_options=self.storage_options,
                input_file_name_col=self.input_file_name_col,
                read_as_dict_cols=self.read_as_dict_cols,
                use_hive=self.use_hive,
            )
        except OSError as e:
            if "non-file path" in str(e):
                raise FileNotFoundError(str(e))
            raise


def get_filters_pyobject(dnf_filter_str, expr_filter_str, vars):  # pragma: no cover
    pass


@overload(get_filters_pyobject, no_unliteral=True)
def overload_get_filters_pyobject(dnf_filter_str, expr_filter_str, var_tup):
    """generate a pyobject for filter expression to pass to C++"""
    dnf_filter_str_val = get_overload_const_str(dnf_filter_str)
    expr_filter_str_val = get_overload_const_str(expr_filter_str)
    var_unpack = ", ".join(f"f{i}" for i in range(len(var_tup)))
    func_text = "def impl(dnf_filter_str, expr_filter_str, var_tup):\n"
    if len(var_tup):
        func_text += f"  {var_unpack}, = var_tup\n"
    func_text += "  with numba.objmode(dnf_filters_py='parquet_predicate_type', expr_filters_py='parquet_predicate_type'):\n"
    func_text += f"    dnf_filters_py = {dnf_filter_str_val}\n"
    func_text += f"    expr_filters_py = {expr_filter_str_val}\n"
    func_text += "  return (dnf_filters_py, expr_filters_py)\n"
    loc_vars = {}
    glbs = globals()
    glbs["numba"] = numba
    exec(func_text, glbs, loc_vars)
    return loc_vars["impl"]


def unify_schemas(schemas):
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
    return pa.unify_schemas(new_schemas)


class ParquetDataset:
    """Stores information about parquet dataset that is needed at compile time
    and runtime (to read the dataset). Stores the list of fragments
    (pieces) that form the dataset and filesystem object to read them.
    All of this is obtained at rank 0 using Arrow's pq.ParquetDataset() API
    (ParquetDatasetV2) and this object is broadcasted to all ranks.
    """

    def __init__(self, pa_pq_dataset, prefix=""):
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
        # Convert large_string Arrow types to string
        # (see comment in bodo.io.parquet_pio.unify_schemas)
        for i in range(len(self.schema)):
            f = self.schema.field(i)
            if f.type == pa.large_string():
                self.schema = self.schema.set(i, f.with_type(pa.string()))
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


class ParquetPiece(object):
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


def get_parquet_dataset(
    fpath,
    get_row_counts: bool = True,
    dnf_filters=None,
    expr_filters=None,
    storage_options=None,
    read_categories: bool = False,
    is_parallel=False,  # only used with get_row_counts=True
    tot_rows_to_read: Optional[int] = None,
    typing_pa_schema: Optional[pa.Schema] = None,
    use_hive: bool = True,
    partitioning="hive",
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
        is_parallel: True if reading in parallel
        tot_rows_to_read: total number of rows to read from dataset. Used at runtime
            for example if doing df.head(tot_rows_to_read) where df is the output of
            read_parquet()
        typing_pa_schema: PyArrow schema determined at compile time. When provided,
            we should validate that the unified schema of all files matches this schema,
            and throw an error otherwise. Currently this is only used in runtime.
            https://bodo.atlassian.net/browse/BE-2787
    """

    if not use_hive:
        partitioning = None

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

    import time

    import pyarrow as pa
    import pyarrow.parquet as pq
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

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
                raise BodoError(
                    "All parquet files must use the same filesystem protocol"
                )
            if u_p.netloc != bucket_name:
                raise BodoError("All parquet files must be in the same S3 bucket")
            fpath[i] = f.rstrip("/")
    else:
        parsed_url = urlparse(fpath)
        protocol = parsed_url.scheme
        fpath = fpath.rstrip("/")

    if protocol in {"gcs", "gs"}:
        try:
            import gcsfs
        except ImportError:
            message = (
                "Couldn't import gcsfs, which is required for Google cloud access."
                " gcsfs can be installed by calling"
                " 'conda install -c conda-forge gcsfs'.\n"
            )
            raise BodoError(message)

    if protocol == "http":
        try:
            import fsspec
        except ImportError:
            message = (
                "Couldn't import fsspec, which is required for http access."
                " fsspec can be installed by calling"
                " 'conda install -c conda-forge fsspec'.\n"
            )

    fs = []

    def getfs(parallel=False):
        # NOTE: add remote filesystems to REMOTE_FILESYSTEMS
        if len(fs) == 1:
            return fs[0]
        if protocol == "s3":
            fs.append(
                get_s3_fs_from_path(
                    fpath, parallel=parallel, storage_options=storage_options
                )
                if not isinstance(fpath, list)
                else get_s3_fs_from_path(
                    fpath[0], parallel=parallel, storage_options=storage_options
                )
            )
        elif protocol in {"gcs", "gs"}:
            # TODO pass storage_options to GCSFileSystem
            google_fs = gcsfs.GCSFileSystem(token=None)
            fs.append(PyFileSystem(FSSpecHandler(google_fs)))
        elif protocol == "http":
            fs.append(PyFileSystem(FSSpecHandler(fsspec.filesystem("http"))))
        elif protocol in {"hdfs", "abfs", "abfss"}:  # pragma: no cover
            fs.append(
                get_hdfs_fs(fpath)
                if not isinstance(fpath, list)
                else get_hdfs_fs(fpath[0])
            )
        else:
            fs.append(pa.fs.LocalFileSystem())
        return fs[0]

    def glob(protocol, fs, path):
        """Return a possibly-empty list of path names that match glob pattern
        given by path"""
        if not protocol and fs is None:
            from fsspec.implementations.local import LocalFileSystem

            fs = LocalFileSystem()
        if isinstance(fs, pa.fs.FileSystem):
            from fsspec.implementations.arrow import ArrowFSWrapper

            fs = ArrowFSWrapper(fs)
        try:
            files = fs.glob(path)
        except:  # pragma: no cover
            raise BodoError(f"glob pattern expansion not supported for {protocol}")
        if len(files) == 0:
            raise BodoError("No files found matching glob pattern")
        return files

    validate_schema = False
    if get_row_counts:
        # Getting row counts and schema validation is going to be
        # distributed across ranks, so every rank will need a filesystem
        # object to query the metadata of their assigned pieces.
        # We have seen issues in the past with broadcasting some filesystem
        # objects (e.g. s3) and even if they don't exist in Arrow >= 8
        # broadcasting the filesystem adds extra time, so instead we initialize
        # the filesystem before the broadcast. That way all ranks do it in parallel
        # at the same time.
        _ = getfs(parallel=True)
        validate_schema = bodo.parquet_validate_schema

    if bodo.get_rank() == 0:
        nthreads = 1  # number of threads to use on rank 0 to collect metadata
        cpu_count = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            nthreads = cpu_count // 2
        try:
            if get_row_counts:
                ev_pq_ds = tracing.Event("pq.ParquetDataset", is_parallel=False)
                if tracing.is_tracing():
                    # only do the work of converting dnf_filters to string
                    # if tracing is enabled
                    ev_pq_ds.add_attribute("g_dnf_filter", str(dnf_filters))
            pa_default_io_thread_count = pa.io_thread_count()
            pa.set_io_thread_count(nthreads)

            prefix = ""
            if protocol == "s3":
                prefix = "s3://"
            elif protocol in {"hdfs", "abfs", "abfss"}:
                # HDFS filesystem is initialized with host:port info. Once
                # initialized, the filesystem needs the <protocol>://<host><port>
                # prefix removed to query and access files
                prefix = f"{protocol}://{parsed_url.netloc}"
            if prefix:
                if isinstance(fpath, list):
                    fpath_noprefix = [f[len(prefix) :] for f in fpath]
                else:
                    fpath_noprefix = fpath[len(prefix) :]
            else:
                fpath_noprefix = fpath

            if isinstance(fpath_noprefix, list):
                # Expand any glob strings in the list in order to generate a
                # single list of fully realized paths to parquet files.
                # For example: ["A/a.pq", "B/*.pq"] might expand to
                # ["A/a.pq", "B/part-0.pq", "B/part-1.pq"]
                new_fpath = []
                for p in fpath_noprefix:
                    if has_magic(p):
                        new_fpath += glob(protocol, getfs(), p)
                    else:
                        new_fpath.append(p)
                fpath_noprefix = new_fpath
            elif has_magic(fpath_noprefix):
                fpath_noprefix = glob(protocol, getfs(), fpath_noprefix)

            dataset = pq.ParquetDataset(
                fpath_noprefix,
                filesystem=getfs(),
                filters=None,  # we pass filters manually below because of Arrow bug
                use_legacy_dataset=False,  # Use ParquetDatasetV2
                partitioning=partitioning,
            )
            if dnf_filters is not None:
                # XXX This is actually done inside _ParquetDatasetV2 constructor,
                # but a bug in Arrow prevents passing compute expression filters.
                # We can remove this and pass the filters to ParquetDataset
                # constructor once the bug is fixed.
                dataset._filters = dnf_filters
                dataset._filter_expression = pq._filters_to_expression(dnf_filters)

            num_files_before_filter = len(dataset.files)
            # If there are dnf_filters files are filtered in ParquetDataset constructor
            dataset = ParquetDataset(dataset, prefix)
            # restore pyarrow default IO thread count
            pa.set_io_thread_count(pa_default_io_thread_count)

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

            if get_row_counts:
                if dnf_filters is not None:
                    ev_pq_ds.add_attribute(
                        "num_pieces_before_filter", num_files_before_filter
                    )
                    ev_pq_ds.add_attribute(
                        "num_pieces_after_filter", len(dataset.pieces)
                    )
                ev_pq_ds.finalize()
        except Exception as e:
            # See note in s3_list_dir_fnames
            # In some cases, OSError/FileNotFoundError can propagate back to numba and comes back as InternalError.
            # where numba errors are hidden from the user.
            # See [BE-1188] for an example
            # Raising a BodoError lets message comes back and seen by the user.
            if isinstance(e, IsADirectoryError):
                # We supress Arrow's error message since it doesn't apply to Bodo
                # (the bit about doing a union of datasets)
                e = BodoError(list_of_files_error_msg)
            elif isinstance(fpath, list) and isinstance(
                e, (OSError, FileNotFoundError)
            ):
                e = BodoError(str(e) + list_of_files_error_msg)
            else:
                e = BodoError(f"error from pyarrow: {type(e).__name__}: {str(e)}\n")
            comm.bcast(e)
            raise e

        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        dataset = comm.bcast(dataset)
    else:
        if get_row_counts:
            ev_bcast = tracing.Event("bcast dataset")
        dataset = comm.bcast(None)
        if isinstance(dataset, Exception):  # pragma: no cover
            error = dataset
            raise error

    # As mentioned above, we don't want to broadcast the filesystem because it
    # adds time (so initially we didn't include it in the dataset). We add
    # it to the dataset now that it's been broadcasted
    dataset.set_fs(getfs())

    if get_row_counts:
        ev_bcast.finalize()

    if get_row_counts and tot_rows_to_read == 0:
        get_row_counts = validate_schema = False
    if get_row_counts or validate_schema:
        # getting row counts and validating schema requires reading
        # the file metadata from the parquet files and is very expensive
        # for datasets consisting of many files, so we do this in parallel
        if get_row_counts and tracing.is_tracing():
            ev_row_counts = tracing.Event("get_row_counts")
            ev_row_counts.add_attribute("g_num_pieces", len(dataset.pieces))
            ev_row_counts.add_attribute("g_expr_filters", str(expr_filters))
        ds_scan_time = 0.0
        num_pieces = len(dataset.pieces)
        start = get_start(num_pieces, bodo.get_size(), bodo.get_rank())
        end = get_end(num_pieces, bodo.get_size(), bodo.get_rank())
        total_rows_chunk = 0
        total_row_groups_chunk = 0
        total_row_groups_size_chunk = 0
        valid = True  # True if schema of all parquet files match
        if expr_filters is not None:
            import random

            random.seed(37)
            pieces = random.sample(dataset.pieces, k=len(dataset.pieces))
        else:
            pieces = dataset.pieces

        fpaths = [p.path for p in pieces[start:end]]
        # Presumably the work is partitioned more or less equally among ranks,
        # and we are mostly (or just) reading metadata, so we assign four IO
        # threads to every rank
        nthreads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), 4)
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
                    # Two files are compatible if arrow can unify their schemas.
                    file_schema = frag.metadata.schema.to_arrow_schema()
                    fileset_schema_names = set(file_schema.names)
                    # Check the names are the same because pa.unify_schemas
                    # will unify a schema where a column is in 1 file but not
                    # another.
                    dataset_schema_names = set(dataset.schema.names) - set(
                        dataset.partition_names
                    )
                    if dataset_schema_names != fileset_schema_names:
                        added_columns = fileset_schema_names - dataset_schema_names
                        missing_columns = dataset_schema_names - fileset_schema_names

                        msg = f"Schema in {piece} was different.\n"
                        if typing_pa_schema is not None:
                            if added_columns:
                                msg += f"File contains column(s) {added_columns} not found in other files in the dataset.\n"
                                raise BodoError(msg)
                        else:
                            if added_columns:
                                msg += f"File contains column(s) {added_columns} not found in other files in the dataset.\n"
                            if missing_columns:
                                msg += f"File missing column(s) {missing_columns} found in other files in the dataset.\n"
                            raise BodoError(msg)
                    try:
                        dataset.schema = unify_schemas([dataset.schema, file_schema])
                    except Exception as e:
                        msg = f"Schema in {piece} was different.\n" + str(e)
                        raise BodoError(msg)

                t0 = time.time()
                row_count = frag.scanner(
                    schema=dataset_.schema,
                    filter=expr_filters,
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

        # synchronize error state
        if comm.allreduce(error is not None, op=MPI.LOR):
            for error in comm.allgather(error):
                if error:
                    if isinstance(fpath, list) and isinstance(
                        error, (OSError, FileNotFoundError)
                    ):
                        raise BodoError(str(error) + list_of_files_error_msg)
                    raise error

        if validate_schema:
            valid = comm.allreduce(valid, op=MPI.LAND)
            if not valid:  # pragma: no cover
                raise BodoError("Schema in parquet files don't match")

        if get_row_counts:
            dataset._bodo_total_rows = comm.allreduce(total_rows_chunk, op=MPI.SUM)
            total_num_row_groups = comm.allreduce(total_row_groups_chunk, op=MPI.SUM)
            total_row_groups_size = comm.allreduce(
                total_row_groups_size_chunk, op=MPI.SUM
            )
            pieces_rows = np.array([p._bodo_num_rows for p in dataset.pieces])
            # communicate row counts to everyone
            pieces_rows = comm.allreduce(pieces_rows, op=MPI.SUM)
            for p, nrows in zip(dataset.pieces, pieces_rows):
                p._bodo_num_rows = nrows
            if (
                is_parallel
                and bodo.get_rank() == 0
                and total_num_row_groups < bodo.get_size()
                and total_num_row_groups != 0
            ):
                warnings.warn(
                    BodoWarning(
                        f"""Total number of row groups in parquet dataset {fpath} ({total_num_row_groups}) is too small for effective IO parallelization.
For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()}). For more details, refer to
https://docs.bodo.ai/latest/file_io/#parquet-section.
"""
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
                    BodoWarning(
                        f"""Parquet average row group size is small ({avg_row_group_size_bytes} bytes) and can have negative impact on performance when reading from remote sources"""
                    )
                )
            if tracing.is_tracing():
                ev_row_counts.add_attribute(
                    "g_total_num_row_groups", total_num_row_groups
                )
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

    if read_categories:
        _add_categories_to_pq_dataset(dataset)

    if get_row_counts:
        ev.finalize()

    if validate_schema:
        if tracing.is_tracing():
            ev_unify_schemas = tracing.Event("unify_schemas_across_ranks")
        error = None

        try:
            dataset.schema = comm.allreduce(
                dataset.schema, bodo.io.helpers.pa_schema_unify_mpi_op
            )
        except Exception as e:
            error = e

        if tracing.is_tracing():
            ev_unify_schemas.finalize()

        # synchronize error state
        if comm.allreduce(error is not None, op=MPI.LOR):
            for error in comm.allgather(error):
                if error:
                    msg = f"Schema in some files were different.\n" + str(error)
                    raise BodoError(msg)

    return dataset


def get_scanner_batches(
    fpaths,
    expr_filters,
    selected_fields,
    avg_num_pieces,
    is_parallel,
    filesystem,
    str_as_dict_cols,
    start_offset,  # starting row offset in the pieces this process is going to read
    rows_to_read,  # total number of rows this process is going to read
    partitioning,
    schema,
):
    """return RecordBatchReader for dataset of 'fpaths' that contain the rows
    that match expr_filters (or all rows if expr_filters is None). Only project the
    fields in selected_fields"""
    import pyarrow as pa

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_threads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), cpu_count)
    max_threads = min(int(os.environ.get("BODO_MAX_IO_THREADS", 16)), cpu_count)
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
    # set columns to be read as dictionary encoded in schema
    dict_col_set = set(str_as_dict_cols)
    for i, name in enumerate(schema.names):
        if name in dict_col_set:
            old_field = schema.field(i)
            new_field = pa.field(
                name, pa.dictionary(pa.int32(), old_field.type), old_field.nullable
            )
            schema = schema.remove(i).insert(i, new_field)
    dataset = ds.dataset(
        fpaths,
        filesystem=filesystem,
        partitioning=partitioning,
        schema=schema,
        format=pq_format,
    )
    col_names = dataset.schema.names
    selected_names = [col_names[field_num] for field_num in selected_fields]

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
    # and this rank needs to skip rows of the first file
    filter_row_groups = len(fpaths) <= 3 or (start_offset > 0 and len(fpaths) <= 10)
    if filter_row_groups and (expr_filters is None):
        # TODO see if getting row counts with filter pushdown could be worthwhile
        # in some specific cases, and integrate that into the above heuristic
        new_frags = []
        # total number of rows of all the row groups we iterate through
        count_rows = 0
        # track total rows that this rank will read from row groups we iterate
        # through
        rows_added = 0
        for frag in dataset.get_fragments():  # each fragment is a parquet file
            # For reference, this is basically the same logic as in
            # ArrowDataframeReader::init_arrow_reader() and just adapted from there.
            # Get the file's row groups that this rank will read from
            row_group_ids = []
            for rg in frag.row_groups:
                num_rows_rg = rg.num_rows
                if start_offset < count_rows + num_rows_rg:
                    # rank needs to read from this row group
                    if rows_added == 0:
                        # this is the first row group the rank will read from
                        start_row_first_rg = start_offset - count_rows
                        rows_added_from_rg = min(
                            num_rows_rg - start_row_first_rg, rows_to_read
                        )
                    else:
                        rows_added_from_rg = min(num_rows_rg, rows_to_read - rows_added)
                    rows_added += rows_added_from_rg
                    row_group_ids.append(rg.id)
                count_rows += num_rows_rg
                if rows_added == rows_to_read:
                    break
            # XXX frag.subset(row_group_ids) is expensive on remote filesystems
            # with datasets composed of many files and row groups
            new_frags.append(frag.subset(row_group_ids=row_group_ids))
            if rows_added == rows_to_read:
                break
        dataset = ds.FileSystemDataset(
            new_frags, dataset.schema, pq_format, filesystem=dataset.filesystem
        )
        # The starting offset the Parquet reader knows about is from the first
        # file, not the first row group, so we need to communicate this back to C++
        start_offset = start_row_first_rg

    reader = dataset.scanner(
        columns=selected_names, filter=expr_filters, use_threads=True
    ).to_reader()
    return dataset, reader, start_offset


# XXX Move this to ParquetDataset class?
def _add_categories_to_pq_dataset(pq_dataset):
    """adds categorical values for each categorical column to the Parquet dataset
    as '_category_info' attribute
    """
    import pyarrow as pa
    from mpi4py import MPI

    # NOTE: shouldn't be possible
    if len(pq_dataset.pieces) < 1:  # pragma: no cover
        raise BodoError(
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


def get_pandas_metadata(schema, num_pieces):
    # find pandas index column if any
    # TODO: other pandas metadata like dtypes needed?
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html
    index_col = None
    # column_name -> is_nullable (or None if unknown)
    nullable_from_metadata = defaultdict(lambda: None)
    key = b"pandas"
    if schema.metadata is not None and key in schema.metadata:
        import json

        pandas_metadata = json.loads(schema.metadata[key].decode("utf8"))
        n_indices = len(pandas_metadata["index_columns"])
        if n_indices > 1:
            raise BodoError("read_parquet: MultiIndex not supported yet")
        index_col = pandas_metadata["index_columns"][0] if n_indices else None
        # ignore non-str/dict index metadata
        if not isinstance(index_col, str) and not isinstance(index_col, dict):
            index_col = None

        for col_dict in pandas_metadata["columns"]:
            col_name = col_dict["name"]
            if col_dict["pandas_type"].startswith("int") and col_name is not None:
                if col_dict["numpy_type"].startswith("Int"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False
    return index_col, nullable_from_metadata


def get_str_columns_from_pa_schema(pa_schema):
    """
    Get the list of string type columns in the schema.
    """
    str_columns = []
    for col_name in pa_schema.names:
        field = pa_schema.field(col_name)
        if field.type in (pa.string(), pa.large_string()):
            str_columns.append(col_name)
    return str_columns


def _pa_schemas_match(pa_schema1, pa_schema2):
    """check if Arrow schemas match or not"""
    # check column names
    if pa_schema1.names != pa_schema2.names:
        return False

    # check type matches
    try:
        unify_schemas([pa_schema1, pa_schema2])
    except:
        return False

    return True


def _get_sample_pq_pieces(pq_dataset, pa_schema, is_iceberg):
    """get a sample of pieces in the Parquet dataset to avoid the overhead of opening
    every file in compile time.

    Also filters the pieces that don't match the table schema, needed for Iceberg to
    avoid errors. We don't know the filters early in the compilation pipeline, so we
    want to avoid schema evolution issues for codes that would otherwise work when
    filters are applied.

    Args:
        pq_dataset (ParquetDataset): input Parquet dataset
        pa_schema (pyarrow.lib.Schema): Arrow schema to check

    Returns:
        list(ParquetPiece): a sample of filtered pieces
    """

    pieces = pq_dataset.pieces

    # a sample of N files where N is the number of ranks. Each rank looks at
    # the metadata of a different random file
    if len(pieces) > bodo.get_size():
        import random

        random.seed(37)
        pieces = random.sample(pieces, bodo.get_size())
    else:
        pieces = pieces

    # Only use pieces that match the target schema. May reduce the sample size in cases
    # with schema evolution, but not very likely to change the outcome due to Iceberg's
    # random file name generation.
    # NOTE: p.metadata opens the Parquet file and can be slow so not filtering before
    # random sampling above.
    # Schema matching isn't straightforward for Parquet datasets, since piece schemas
    # don't include the Hive partitioned columns.
    # See https://bodo.atlassian.net/browse/BE-3679
    if is_iceberg:
        pieces = [
            p
            for p in pieces
            if _pa_schemas_match(p.metadata.schema.to_arrow_schema(), pa_schema)
        ]

    return pieces


def determine_str_as_dict_columns(
    pq_dataset, pa_schema, str_columns: list, is_iceberg: bool = False
) -> set:
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

    # get a sample of Parquet pieces to avoid opening every file in compile time
    pieces = _get_sample_pq_pieces(pq_dataset, pa_schema, is_iceberg)

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
    use_hive=True,
) -> FileSchema:
    """get parquet schema from file using Parquet dataset and Arrow APIs"""
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
        use_hive=use_hive,
    )

    partition_names = pq_dataset.partition_names
    pa_schema = pq_dataset.schema
    num_pieces = len(pq_dataset.pieces)

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
                bodo.utils.typing.BodoWarning,
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
    # index's name if there is one, otherwise "__index__level_0"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names
    index_col, nullable_from_metadata = get_pandas_metadata(pa_schema, num_pieces)
    col_types_total = []
    is_supported_list = []
    arrow_types = []
    for i, c in enumerate(col_names):
        if c in partition_names:
            continue
        field = pa_schema.field(c)
        dtype, supported = _get_numba_typ_from_pa_typ(
            field,
            c == index_col,
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
            raise BodoError(f"Selected column {c} not in Parquet file schema")
    if (
        index_col
        and not isinstance(index_col, dict)
        and index_col not in selected_columns
    ):
        # if index_col is "__index__level_0" or some other name, append it.
        # If the index column is not selected when reading parquet, the index
        # should still be included.
        selected_columns.append(index_col)

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
        index_col,
        col_indices,
        partition_names,
        unsupported_columns,
        unsupported_arrow_types,
        pa_schema,
    )


def _get_partition_cat_dtype(dictionary):
    """get categorical dtype for Parquet partition set"""
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


# ------------------------- Parquet Write C++ ------------------------- #
from llvmlite import ir as lir
from numba.core import cgutils

if bodo.utils.utils.has_pyarrow():
    from bodo.io import arrow_cpp

    ll.add_symbol("pq_write", arrow_cpp.pq_write)
    ll.add_symbol("pq_write_partitioned", arrow_cpp.pq_write_partitioned)


@intrinsic
def parquet_write_table_cpp(
    typingctx,
    filename_t,
    table_t,
    col_names_t,
    index_t,
    write_index,
    metadata_t,
    compression_t,
    is_parallel_t,
    write_range_index,
    start,
    stop,
    step,
    name,
    bucket_region,
    row_group_size,
    file_prefix,
    convert_timedelta_to_int64,
    timestamp_tz,
    downcast_time_ns_to_us,
):
    """
    Call C++ parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),  # convert_timedelta_to_int64
                lir.IntType(8).as_pointer(),  # tz
                lir.IntType(1),  # downcast_time_ns_to_us
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="pq_write")
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.int64(
            types.voidptr,
            table_t,
            col_names_t,
            index_t,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.int32,
            types.int32,
            types.int32,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.boolean,  # convert_timedelta_to_int64
            types.voidptr,  # tz
            types.boolean,  # downcast_time_ns_to_us
        ),
        codegen,
    )


@intrinsic
def parquet_write_table_partitioned_cpp(
    typingctx,
    filename_t,
    data_table_t,
    col_names_t,
    col_names_no_partitions_t,
    cat_table_t,
    part_col_idxs_t,
    num_part_col_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    file_prefix,
    timestamp_tz,
):
    """
    Call C++ parquet write partitioned function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),  # tz
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="pq_write_partitioned"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.void(
            types.voidptr,
            data_table_t,
            col_names_t,
            col_names_no_partitions_t,
            cat_table_t,
            types.voidptr,
            types.int32,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.voidptr,  # tz
        ),
        codegen,
    )
