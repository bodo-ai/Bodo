# Copyright (C) 2019 Bodo Inc. All rights reserved.
import warnings
import numba
from numba.core import ir, types
from numba.core.ir_utils import (
    mk_unique_var,
    find_const,
    compile_to_numba_ir,
    replace_arg_nodes,
    guard,
)

from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core.imputils import impl_ret_new_ref
import numpy as np
import bodo
from bodo.libs.str_ext import string_type, unicode_to_char_ptr
from bodo.libs.str_arr_ext import (
    StringArrayPayloadType,
    construct_string_array,
    string_array_type,
)
from bodo.libs.list_str_arr_ext import (
    list_string_array_type,
    ListStringArrayPayloadType,
    construct_list_string_array,
)
from bodo.hiframes.datetime_date_ext import (
    datetime_date_type,
    datetime_date_array_type,
    DatetimeDateArrayType,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType, Decimal128Type
from bodo.libs.bool_arr_ext import boolean_array, BooleanArrayType
from bodo.utils.utils import unliteral_all, sanitize_varname
from bodo.utils.typing import BodoError, BodoWarning
import bodo.ir.parquet_ext
from bodo.transforms import distributed_pass
from bodo.utils.transform import get_str_const_value
from numba.extending import intrinsic
from collections import defaultdict


# read Arrow Int columns as nullable int array (IntegerArrayType)
use_nullable_int_arr = True


def read_parquet():  # pragma: no cover
    return 0


def read_parquet_str():  # pragma: no cover
    return 0


def read_parquet_list_str():  # pragma: no cover
    return 0


def get_column_size_parquet():  # pragma: no cover
    return 0


class ParquetHandler:
    """analyze and transform parquet IO calls"""

    def __init__(self, func_ir, typingctx, args, _locals, _reverse_copies):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.args = args
        self.locals = _locals
        self.reverse_copies = _reverse_copies

    def gen_parquet_read(self, file_name, lhs, columns):
        scope = lhs.scope
        loc = lhs.loc

        table_types = None
        # lhs is temporary and will possibly be assigned to user variable
        assert lhs.name.startswith("$")
        if (
            lhs.name in self.reverse_copies
            and self.reverse_copies[lhs.name] in self.locals
        ):
            table_types = self.locals[self.reverse_copies[lhs.name]]
            self.locals.pop(self.reverse_copies[lhs.name])

        convert_types = {}
        # user-specified type conversion
        if (
            lhs.name in self.reverse_copies
            and (self.reverse_copies[lhs.name] + ":convert") in self.locals
        ):
            convert_types = self.locals[self.reverse_copies[lhs.name] + ":convert"]
            self.locals.pop(self.reverse_copies[lhs.name] + ":convert")

        if table_types is None:
            msg = (
                "Parquet schema not available. Either path "
                "argument should be constant for Bodo to look at the file "
                "at compile time or schema should be provided."
            )
            file_name_str = get_str_const_value(
                file_name, self.func_ir, msg, arg_types=self.args
            )
            col_names, col_types, index_col, col_indices = parquet_file_schema(
                file_name_str, columns
            )
        else:
            col_names = list(table_types.keys())
            col_types = [t for t in table_types.values()]
            index_col = "index" if "index" in col_names else None
            col_indices = list(range(len(col_names)))
            # TODO: allow specifying types of only selected columns
            if columns is not None:
                col_indices = [col_names.index(c) for c in columns]
                col_types = [col_types[i] for i in col_indices]
                col_names = columns
                index_col = index_col if index_col in col_names else None

        # HACK convert types using decorator for int columns with NaN
        for i, c in enumerate(col_names):
            if c in convert_types:
                col_types[i] = convert_types[c]

        data_arrs = [ir.Var(scope, mk_unique_var(c), loc) for c in col_names]
        nodes = [
            bodo.ir.parquet_ext.ParquetReader(
                file_name, lhs.name, col_names, col_indices, col_types, data_arrs, loc
            )
        ]
        return col_names, data_arrs, index_col, nodes


def pq_distributed_run(
    pq_node, array_dists, typemap, calltypes, typingctx, targetctx, dist_pass
):

    n_cols = len(pq_node.out_vars)
    # get column variables and their sizes
    arg_names = ", ".join("out" + str(i) for i in range(2 * n_cols))
    func_text = "def pq_impl(fname):\n"
    func_text += "    ({},) = _pq_reader_py(fname)\n".format(arg_names)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    pq_impl = loc_vars["pq_impl"]

    # parallel columns
    parallel = [
        c
        for c, v in zip(pq_node.col_names, pq_node.out_vars)
        if array_dists[v.name]
        in (distributed_pass.Distribution.OneD, distributed_pass.Distribution.OneD_Var)
    ]

    pq_reader_py = _gen_pq_reader_py(
        pq_node.col_names,
        pq_node.col_indices,
        pq_node.out_types,
        typingctx,
        targetctx,
        parallel,
    )

    f_block = compile_to_numba_ir(
        pq_impl,
        {"_pq_reader_py": pq_reader_py},
        typingctx,
        (string_type,),
        typemap,
        calltypes,
    ).blocks.popitem()[1]
    replace_arg_nodes(f_block, [pq_node.file_name])
    nodes = f_block.body[:-3]

    for i in range(n_cols):
        nodes[2 * (i - n_cols)].target = pq_node.out_vars[i]

    return nodes


distributed_pass.distributed_run_extensions[
    bodo.ir.parquet_ext.ParquetReader
] = pq_distributed_run


def _gen_pq_reader_py(
    col_names, col_indices, out_types, typingctx, targetctx, parallel
):
    if len(parallel) > 0:
        # in parallel read, we assume all columns are parallel
        assert col_names == parallel
    is_parallel = len(parallel) > 0
    func_text = "def pq_reader_py(fname):\n"
    # open a DatasetReader, which is a C++ object defined in _parquet.cpp that
    # contains file readers for the files from which this process needs to read,
    # and other information to read this process' chunk
    func_text += "  ds_reader = get_dataset_reader(unicode_to_char_ptr(fname), {})\n".format(
        is_parallel
    )
    for c, ind, t in zip(col_names, col_indices, out_types):
        func_text = gen_column_read(func_text, c, ind, t, c in parallel)
    func_text += "  del_dataset_reader(ds_reader)\n"
    func_text += "  return ({},)\n".format(
        ", ".join("{0}, {0}_size".format(sanitize_varname(c)) for c in col_names)
    )

    loc_vars = {}
    glbs = {
        "get_dataset_reader": _get_dataset_reader,
        "del_dataset_reader": _del_dataset_reader,
        "get_column_size_parquet": get_column_size_parquet,
        "read_parquet": read_parquet,
        "read_parquet_str": read_parquet_str,
        "read_parquet_list_str": read_parquet_list_str,
        "unicode_to_char_ptr": unicode_to_char_ptr,
        "NS_DTYPE": np.dtype("M8[ns]"),
        "np": np,
        "bodo": bodo,
    }

    exec(func_text, glbs, loc_vars)
    pq_reader_py = loc_vars["pq_reader_py"]

    jit_func = numba.njit(pq_reader_py, no_cpython_wrapper=True)
    return jit_func


def gen_column_read(func_text, cname, c_ind, c_type, is_parallel):
    cname = sanitize_varname(cname)
    # handle size variables
    # get_column_size_parquet returns local size of column (number of rows
    # that this process is going to read)
    func_text += "  {}_size = get_column_size_parquet(ds_reader, {})\n".format(
        cname, c_ind
    )
    alloc_size = "{}_size".format(cname)

    if c_type == string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += "  {} = read_parquet_str(ds_reader, {}, {}_size)\n".format(
            cname, c_ind, cname
        )
    elif c_type == list_string_array_type:
        # pass size for easier allocation and distributed analysis
        func_text += "  {} = read_parquet_list_str(ds_reader, {}, {}_size)\n".format(
            cname, c_ind, cname
        )
    else:
        el_type = get_element_type(c_type.dtype)
        func_text += _gen_alloc(c_type, cname, alloc_size, el_type)
        func_text += "  status = read_parquet(ds_reader, {}, {}, np.int32({}))\n".format(
            c_ind, cname, bodo.utils.utils.numba_to_c_type(c_type.dtype)
        )

    return func_text


def _gen_alloc(c_type, cname, alloc_size, el_type):
    if isinstance(c_type, IntegerArrayType):
        return "  {0} = bodo.libs.int_arr_ext.alloc_int_array({1}, {2})\n".format(
            cname, alloc_size, el_type
        )
    if c_type == boolean_array:
        return "  {0} = bodo.libs.bool_arr_ext.alloc_bool_array({1})\n".format(
            cname, alloc_size
        )
    if isinstance(c_type, DecimalArrayType):
        return "  {0} = bodo.libs.decimal_arr_ext.alloc_decimal_array({1}, {2}, {3})\n".format(
            cname, alloc_size, c_type.precision, c_type.scale
        )
    if c_type == datetime_date_array_type:
        return "  {0} = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array({1})\n".format(
            cname, alloc_size
        )
    return "  {} = np.empty({}, dtype={})\n".format(cname, alloc_size, el_type)


def get_element_type(dtype):
    """get dtype string to pass to empty() allocations
    """
    if dtype == types.NPDatetime("ns"):
        # NS_DTYPE has to be defined in function globals
        return "NS_DTYPE"

    if isinstance(dtype, Decimal128Type):
        return "int128"

    if dtype == datetime_date_type:
        return "np.int32"

    out = repr(dtype)
    if out == "bool":  # fix bool string
        out = "bool_"

    return "np." + out


def _get_numba_typ_from_pa_typ(pa_typ, is_index, nullable_from_metadata):
    import pyarrow as pa

    # TODO: comparing list(string) type using pa_typ.type == pa.list_(pa.string())
    # doesn't seem to work properly. The string representation is also inconsistent:
    # "ListType(list<element: string>)", or "ListType(list<item: string>)"
    # likely an Arrow/Parquet bug
    if isinstance(pa_typ.type, pa.ListType) and pa_typ.type.value_type == pa.string():
        return list_string_array_type

    # Decimal128Array type
    if isinstance(pa_typ.type, pa.Decimal128Type):
        return DecimalArrayType(pa_typ.type.precision, pa_typ.type.scale)

    _typ_map = {
        # boolean
        pa.bool_(): types.bool_,
        # signed int types
        pa.int8(): types.int8,
        pa.int16(): types.int16,
        pa.int32(): types.int32,
        pa.int64(): types.int64,
        # unsigned int types
        pa.uint8(): types.uint8,
        pa.uint16(): types.uint16,
        pa.uint32(): types.uint32,
        pa.uint64(): types.uint64,
        # float types (TODO: float16?)
        pa.float32(): types.float32,
        pa.float64(): types.float64,
        # String
        pa.string(): string_type,
        # date
        pa.date32(): datetime_date_type,
        pa.date64(): types.NPDatetime("ns"),
        # time (TODO: time32, time64, ...)
        pa.timestamp("ns"): types.NPDatetime("ns"),
        pa.timestamp("us"): types.NPDatetime("ns"),
        pa.timestamp("ms"): types.NPDatetime("ns"),
        pa.timestamp("s"): types.NPDatetime("ns"),
    }
    if pa_typ.type not in _typ_map:
        raise BodoError("Arrow data type {} not supported yet".format(pa_typ.type))
    dtype = _typ_map[pa_typ.type]

    if dtype == datetime_date_type:
        return datetime_date_array_type

    arr_typ = string_array_type if dtype == string_type else types.Array(dtype, 1, "C")

    if dtype == types.bool_:
        arr_typ = boolean_array

    if nullable_from_metadata is not None:
        # do what metadata says
        _use_nullable_int_arr = nullable_from_metadata
    else:
        # use our global default
        _use_nullable_int_arr = use_nullable_int_arr
    # TODO: support nullable int for indices
    if (
        _use_nullable_int_arr
        and not is_index
        and isinstance(dtype, types.Integer)
        and pa_typ.nullable
    ):
        arr_typ = IntegerArrayType(dtype)

    return arr_typ


def get_parquet_dataset(file_name, parallel, get_row_counts=True):
    """ Create ParquetDataset instance from Parquet file name
        parallel: if true only rank 0 reads dataset and broadcasts to others
        get_row_counts : get row counts of pieces from metadata and store
                         as attributes in ParquetDataset object
    """
    import pyarrow.parquet as pq
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # we read the dataset info only on rank 0 and broadcast it to the rest
    if parallel and bodo.get_rank() != 0:  # pragma: no cover
        dataset = comm.bcast(None)
        return dataset

    fs = None
    dataset = None
    if file_name.startswith("s3://"):
        try:
            import s3fs
        except:
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
        ):
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

        file_names = None
        try:
            # check if file_name is a directory, and if there is a zero-size object
            # with the name of the directory. If there is, we have to omit it
            # because pq.ParquetDataset will throw Invalid Parquet file size is 0
            # bytes
            path_info = fs.info(file_name)
            if (
                path_info["Size"] == 0 and path_info["type"] == "directory"
            ):  # pragma: no cover
                # excluded from coverage because haven't found a reliable way
                # to create 0 size object that is a directory. For example:
                # fs.mkdir(path)  sometimes doesn't do anything at all
                path = file_name  # this is "s3://bucket/path-to-dir"
                files = fs.ls(path)
                if (
                    files
                    and (files[0] == path[5:] or files[0] == path[5:] + "/")
                    and fs.info("s3://" + files[0])["Size"] == 0
                ):
                    # get actual names of objects inside the dir
                    file_names = ["s3://" + fname for fname in files[1:]]
        except:  # pragma: no cover
            pass
        if file_names is not None:  # pragma: no cover
            try:
                dataset = pq.ParquetDataset(file_names, filesystem=fs)
            except Exception as e:
                raise BodoError(
                    "read_parquet(): S3 file system cannot be created: {}".format(e)
                )
    elif file_name.startswith("hdfs://"):  # pragma: no cover
        # this HadoopFileSystem is the new file system of pyarrow
        from pyarrow.fs import HadoopFileSystem, FileSelector, FileType

        # this HadoopFileSystem is the deprecated file system of pyarrow
        # need this for pq.ParquetDataset
        # because the new HadoopFileSystem is not a subclass of
        # pyarrow.filesystem.FileSystem which causes an error
        from pyarrow.hdfs import HadoopFileSystem as HdFS

        # creates a new Hadoop file system from uri
        try:
            hdfs = HadoopFileSystem.from_uri(file_name)
            from urllib.parse import urlparse

            options = urlparse(file_name)
            path = options.path
            fs = HdFS(host=options.hostname, port=options.port, user=options.username)
        except Exception as e:
            raise BodoError(
                "read_parquet(): Hadoop file system cannot be created: {}".format(e)
            )

        # prefix in form of hdfs://host:port
        prefix = file_name[: len(file_name) - len(path)]
        file_names = None
        # target stat of the path: file or just the directory itself
        target_stat = hdfs.get_file_info([file_name])

        if target_stat[0].type in (FileType.NotFound, FileType.Unknown):
            raise BodoError(
                "read_parquet(): {} is a "
                "non-existing or unreachable file".format(file_name)
            )

        if (not target_stat[0].size) and target_stat[0].type == FileType.Directory:
            file_selector = FileSelector(path, recursive=True)
            try:
                file_stats = hdfs.get_file_info(file_selector)
            except Exception as e:
                raise BodoError(
                    "read_parquet(): Exception on getting directory info "
                    "of {}: {}".format(path, e)
                )
            for file_stat in file_stats:
                file_names = [prefix + file_stat.path for file_stat in file_stats]

        if file_names is not None:
            dataset = pq.ParquetDataset(file_names, filesystem=fs)

    if dataset is None:
        dataset = pq.ParquetDataset(file_name, filesystem=fs)

    # store the total number of rows and rows of each piece in the ParquetDataset
    # object, then broadcast to every process
    # NOTE: the information that other processes need to only open file
    # readers for their chunk are the total number of rows and number of rows
    # of each piece, as well as the path of each piece
    if get_row_counts:
        dataset._bodo_total_rows = 0
        for piece in dataset.pieces:
            piece._bodo_num_rows = piece.get_metadata().num_rows
            dataset._bodo_total_rows += piece._bodo_num_rows
    if parallel:
        assert bodo.get_rank() == 0
        comm.bcast(dataset)
    return dataset


def parquet_file_schema(file_name, selected_columns):
    """get parquet schema from file using Parquet dataset and Arrow APIs
    """
    col_names = []
    col_types = []

    # during compilation we only need the schema and it has to be the same for
    # all processes, so we can set parallel=True to just have rank 0 read
    # the dataset information and broadcast to others
    pq_dataset = get_parquet_dataset(file_name, parallel=True, get_row_counts=False)
    pa_schema = pq_dataset.schema.to_arrow_schema()
    num_pieces = len(pq_dataset.pieces)

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with
    # names of list types columns (arrow 0.17.0)
    # col_names is an array that contains all the column's name and
    # index's name if there is one, otherwise "__index__level_0"
    # If there is no index at all, col_names will not include anything.
    col_names = pa_schema.names

    # find pandas index column if any
    # TODO: other pandas metadata like dtypes needed?
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html
    index_col = None
    # column_name -> is_nullable (or None if unknown)
    nullable_from_metadata = defaultdict(lambda: None)
    key = b"pandas"
    if pa_schema.metadata is not None and key in pa_schema.metadata:
        import json

        pandas_metadata = json.loads(pa_schema.metadata[key].decode("utf8"))
        n_indices = len(pandas_metadata["index_columns"])
        if n_indices > 1:
            raise BodoError("read_parquet: MultiIndex not supported yet")
        index_col = pandas_metadata["index_columns"][0] if n_indices else None
        # ignore RangeIndex metadata for multi-part datasets
        # TODO: check what pandas/pyarrow does
        if not isinstance(index_col, str) and (
            not isinstance(index_col, dict) or num_pieces != 1
        ):
            index_col = None

        for col_dict in pandas_metadata["columns"]:
            col_name = col_dict["name"]
            if col_dict["pandas_type"].startswith("int"):
                if col_dict["numpy_type"].startswith("Int"):
                    nullable_from_metadata[col_name] = True
                else:
                    nullable_from_metadata[col_name] = False

    # handle column selection if available
    col_indices = list(range(len(col_names)))
    if selected_columns is not None:
        # make sure selected columns are in the schema
        for c in selected_columns:
            if c not in col_names:
                raise BodoError(
                    "Selected column {} not in Parquet file schema".format(c)
                )
        if index_col and not isinstance(index_col, dict):
            # if index_col is "__index__level_0" or some other name, append it.
            # If the index column is not selected when reading parquet, the index
            # should still be included.
            selected_columns.append(index_col)
        col_indices = [col_names.index(c) for c in selected_columns]
        col_names = selected_columns

    col_types = [
        _get_numba_typ_from_pa_typ(
            pa_schema.field(c), c == index_col, nullable_from_metadata[c]
        )
        for c in col_names
    ]
    # TODO: close file?
    return col_names, col_types, index_col, col_indices


_get_dataset_reader = types.ExternalFunction(
    "get_dataset_reader", types.Opaque("arrow_reader")(types.voidptr, types.boolean)
)
_del_dataset_reader = types.ExternalFunction(
    "del_dataset_reader", types.void(types.Opaque("arrow_reader"))
)


@infer_global(get_column_size_parquet)
class SizeParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.intp, args[0], types.unliteral(args[1]))


@infer_global(read_parquet)
class ReadParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        if args[2] == types.intp:  # string read call, returns string array
            return signature(string_array_type, *unliteral_all(args))
        return signature(types.int64, *unliteral_all(args))


@infer_global(read_parquet_str)
class ReadParquetStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(string_array_type, *unliteral_all(args))


@infer_global(read_parquet_list_str)
class ReadParquetListStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(list_string_array_type, *unliteral_all(args))


from numba.core import cgutils
from numba.core.imputils import lower_builtin
from numba.np.arrayobj import make_array
from llvmlite import ir as lir
import llvmlite.binding as ll

from bodo.config import _has_pyarrow

if _has_pyarrow:
    from bodo.io import parquet_cpp

    ll.add_symbol("get_dataset_reader", parquet_cpp.get_dataset_reader)
    ll.add_symbol("del_dataset_reader", parquet_cpp.del_dataset_reader)
    ll.add_symbol("pq_get_size", parquet_cpp.get_size)
    ll.add_symbol("pq_read", parquet_cpp.read)
    ll.add_symbol("pq_read_string", parquet_cpp.read_string)
    ll.add_symbol("pq_read_list_string", parquet_cpp.read_list_string)
    ll.add_symbol("pq_write", parquet_cpp.pq_write)


@lower_builtin(get_column_size_parquet, types.Opaque("arrow_reader"), types.intp)
def pq_size_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = builder.module.get_or_insert_function(fnty, name="pq_get_size")
    return builder.call(fn, args)


@lower_builtin(
    read_parquet, types.Opaque("arrow_reader"), types.intp, types.Array, types.int32
)
def pq_read_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(8).as_pointer(),
        ],
    )
    out_array = make_array(sig.args[2])(context, builder, args[2])
    zero_ptr = context.get_constant_null(types.voidptr)

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    return builder.call(
        fn,
        [
            args[0],
            args[1],
            builder.bitcast(out_array.data, lir.IntType(8).as_pointer()),
            args[3],
            zero_ptr,
        ],
    )


########################## read nullable int array ###########################


@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    IntegerArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    BooleanArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    DecimalArrayType,
    types.int32,
)
@lower_builtin(
    read_parquet,
    types.Opaque("arrow_reader"),
    types.intp,
    datetime_date_array_type,
    types.int32,
)
def pq_read_int_arr_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(64),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(8).as_pointer(),
        ],
    )
    int_arr_typ = sig.args[2]
    int_arr = cgutils.create_struct_proxy(int_arr_typ)(context, builder, args[2])
    dtype = int_arr_typ.dtype
    if isinstance(int_arr_typ, DecimalArrayType):
        dtype = bodo.libs.decimal_arr_ext.int128_type
    if int_arr_typ == datetime_date_array_type:
        dtype = types.int64
    data_typ = types.Array(dtype, 1, "C")
    data_array = make_array(data_typ)(context, builder, int_arr.data)
    null_arr_typ = types.Array(types.uint8, 1, "C")
    bitmap = make_array(null_arr_typ)(context, builder, int_arr.null_bitmap)

    fn = builder.module.get_or_insert_function(fnty, name="pq_read")
    return builder.call(
        fn,
        [
            args[0],
            args[1],
            builder.bitcast(data_array.data, lir.IntType(8).as_pointer()),
            args[3],
            builder.bitcast(bitmap.data, lir.IntType(8).as_pointer()),
        ],
    )


############################## read strings ###############################


@lower_builtin(read_parquet_str, types.Opaque("arrow_reader"), types.intp, types.intp)
def pq_read_string_lower(context, builder, sig, args):

    typ = sig.return_type
    dtype = StringArrayPayloadType()
    meminfo, meminfo_data_ptr = construct_string_array(context, builder)
    string_array = context.make_helper(builder, typ)

    str_arr_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    str_arr_payload.num_strings = args[2]

    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_string")
    res = builder.call(
        fn,
        [
            args[0],
            args[1],
            str_arr_payload._get_ptr_by_name("offsets"),
            str_arr_payload._get_ptr_by_name("data"),
            str_arr_payload._get_ptr_by_name("null_bitmap"),
        ],
    )
    builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)

    string_array.meminfo = meminfo
    ret = string_array._getvalue()
    return impl_ret_new_ref(context, builder, typ, ret)


############################## read list of strings ###############################


@lower_builtin(
    read_parquet_list_str, types.Opaque("arrow_reader"), types.intp, types.intp
)
def pq_read_list_string_lower(context, builder, sig, args):

    # construct array and payload
    typ = sig.return_type
    dtype = ListStringArrayPayloadType()
    meminfo, meminfo_data_ptr = construct_list_string_array(context, builder)
    list_str_array = context.make_helper(builder, typ)

    list_str_arr_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    list_str_array.num_items = args[2]  # set size

    # read payload data
    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_list_string")
    _res = builder.call(
        fn,
        [
            args[0],
            args[1],
            list_str_arr_payload._get_ptr_by_name("data_offsets"),
            list_str_arr_payload._get_ptr_by_name("index_offsets"),
            list_str_arr_payload._get_ptr_by_name("data"),
            list_str_arr_payload._get_ptr_by_name("null_bitmap"),
        ],
    )

    # set array values
    builder.store(list_str_arr_payload._getvalue(), meminfo_data_ptr)
    list_str_array.meminfo = meminfo
    list_str_array.data_offsets = list_str_arr_payload.data_offsets
    list_str_array.index_offsets = list_str_arr_payload.index_offsets
    list_str_array.data = list_str_arr_payload.data
    list_str_array.null_bitmap = list_str_arr_payload.null_bitmap
    list_str_array.num_total_strings = builder.zext(
        builder.load(
            builder.gep(list_str_array.index_offsets, [list_str_array.num_items])
        ),
        lir.IntType(64),
    )
    list_str_array.num_total_chars = builder.zext(
        builder.load(
            builder.gep(list_str_array.data_offsets, [list_str_array.num_total_strings])
        ),
        lir.IntType(64),
    )
    ret = list_str_array._getvalue()
    return impl_ret_new_ref(context, builder, typ, ret)


############################ parquet write table #############################


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
):
    """
    Call C++ parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
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
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="pq_write")
        builder.call(fn_tp, args)

    return (
        types.void(
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
        ),
        codegen,
    )
