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


def read_parquet_str_parallel():  # pragma: no cover
    return 0


def read_parquet_list_str():  # pragma: no cover
    return 0


def read_parquet_list_str_parallel():  # pragma: no cover
    return 0


def read_parquet_parallel():  # pragma: no cover
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
    # print(func_text)

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

    func_text = "def pq_reader_py(fname):\n"
    func_text += "  arrow_readers = get_arrow_readers(unicode_to_char_ptr(fname))\n"
    for c, ind, t in zip(col_names, col_indices, out_types):
        func_text = gen_column_read(func_text, c, ind, t, c in parallel)
    func_text += "  del_arrow_readers(arrow_readers)\n"
    func_text += "  return ({},)\n".format(
        ", ".join("{0}, {0}_size".format(sanitize_varname(c)) for c in col_names)
    )

    loc_vars = {}
    glbs = {
        "get_arrow_readers": _get_arrow_readers,
        "del_arrow_readers": _del_arrow_readers,
        "get_column_size_parquet": get_column_size_parquet,
        "read_parquet": read_parquet,
        "read_parquet_parallel": read_parquet_parallel,
        "read_parquet_str": read_parquet_str,
        "read_parquet_str_parallel": read_parquet_str_parallel,
        "read_parquet_list_str": read_parquet_list_str,
        "read_parquet_list_str_parallel": read_parquet_list_str_parallel,
        "get_start_count": bodo.libs.distributed_api.get_start_count,
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
    func_text += "  {}_size = get_column_size_parquet(arrow_readers, {})\n".format(
        cname, c_ind
    )
    if is_parallel:
        func_text += "  {0}_start, {0}_count = get_start_count({0}_size)\n".format(
            cname
        )
        alloc_size = "{}_count".format(cname)
    else:
        alloc_size = "{}_size".format(cname)

    # generate strings differently
    if c_type == string_array_type:
        if is_parallel:
            func_text += "  {0} = read_parquet_str_parallel(arrow_readers, {1}, {0}_start, {0}_count)\n".format(
                cname, c_ind
            )
        else:
            # pass size for easier allocation and distributed analysis
            func_text += "  {} = read_parquet_str(arrow_readers, {}, {}_size)\n".format(
                cname, c_ind, cname
            )
    elif c_type == list_string_array_type:
        if is_parallel:
            func_text += "  {0} = read_parquet_list_str_parallel(arrow_readers, {1}, {0}_start, {0}_count)\n".format(
                cname, c_ind
            )
        else:
            # pass size for easier allocation and distributed analysis
            func_text += "  {} = read_parquet_list_str(arrow_readers, {}, {}_size)\n".format(
                cname, c_ind, cname
            )
    else:
        el_type = get_element_type(c_type.dtype)
        func_text += _gen_alloc(c_type, cname, alloc_size, el_type)
        if is_parallel:
            func_text += "  status = read_parquet_parallel(arrow_readers, {0}, {1}, np.int32({2}), {1}_start, {1}_count)\n".format(
                c_ind, cname, bodo.utils.utils.numba_to_c_type(c_type.dtype)
            )
        else:
            func_text += "  status = read_parquet(arrow_readers, {}, {}, np.int32({}))\n".format(
                c_ind, cname, bodo.utils.utils.numba_to_c_type(c_type.dtype)
            )

    return func_text


def _gen_alloc(c_type, cname, alloc_size, el_type):
    if isinstance(c_type, IntegerArrayType):
        return "  {0} = bodo.libs.int_arr_ext.init_integer_array(np.empty({1}, {2}), np.empty(({1} + 7) >> 3, np.uint8))\n".format(
            cname, alloc_size, el_type
        )
    if c_type == boolean_array:
        return "  {0} = bodo.libs.bool_arr_ext.init_bool_array(np.empty({1}, {2}), np.empty(({1} + 7) >> 3, np.uint8))\n".format(
            cname, alloc_size, el_type
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


def get_parquet_dataset(file_name):
    """create ParquetDataset instance from Parquet file name
    """
    import pyarrow.parquet as pq

    fs = None
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
                ParquetDataset = pq.ParquetDataset(file_names, filesystem=fs)
            except Exception as e:
                raise BodoError(
                    "read_parquet(): S3 file system cannot be created: {}".format(e)
                )
            return ParquetDataset
    elif file_name.startswith("hdfs://"):
        # this HadoopFileSystem is the new file system of pyarrow
        from pyarrow.fs import HadoopFileSystem, FileSelector, FileType, HdfsOptions

        # this HadoopFileSystem is the deprecated file system of pyarrow
        # need this for pq.ParquetDataset
        # because the new HadoopFileSystem is not a subclass of
        # pyarrow.filesystem.FileSystem which causes an error
        from pyarrow.hdfs import HadoopFileSystem as HdFS

        # creates a new Hadoop file system from uri
        try:
            hdfs, path = HadoopFileSystem.from_uri(file_name)
            hdfs_options = HdfsOptions.from_uri(file_name)
            (host, port) = hdfs_options.endpoint
            host = host[7:]  # remove hdfs:// prefix
            fs = HdFS(host=host, port=port, user=hdfs_options.user)
        except Exception as e:
            raise BodoError(
                "read_parquet(): Hadoop file system cannot be created: {}".format(e)
            )

        # prefix in form of hdfs://host:port
        prefix = file_name[: len(file_name) - len(path)]
        file_names = None
        # target stat of the path: file or just the directory itself
        target_stat = hdfs.get_target_stats([path])

        if target_stat[0].type == FileType.NonExistent:
            raise BodoError(
                "read_parquet(): {} is a "
                "non-existing or unreachable file".format(file_name)
            )

        if (not target_stat[0].size) and target_stat[0].type == FileType.Directory:
            file_selector = FileSelector(path, allow_non_existent=False, recursive=True)
            try:
                file_stats = hdfs.get_target_stats(file_selector)
            except Exception as e:
                raise BodoError(
                    "read_parquet(): Exception on getting target stats "
                    "of {}: {}".format(path, e)
                )
            for file_stat in file_stats:
                file_names = [prefix + file_stat.path for file_stat in file_stats]

        if file_names is not None:
            return pq.ParquetDataset(file_names, filesystem=fs)

    return pq.ParquetDataset(file_name, filesystem=fs)


def parquet_file_schema(file_name, selected_columns):
    """get parquet schema from file using Parquet dataset and Arrow APIs
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    col_names = []
    col_types = []

    num_pieces = None
    pa_schema = None
    if bodo.get_rank() == 0:
        pq_dataset = get_parquet_dataset(file_name)
        pa_schema = pq_dataset.schema.to_arrow_schema()
        num_pieces = len(pq_dataset.pieces)

    pa_schema, num_pieces = comm.bcast((pa_schema, num_pieces))

    # NOTE: use arrow schema instead of the dataset schema to avoid issues with names of
    # list types columns (arrow 0.16.0)
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


_get_arrow_readers = types.ExternalFunction(
    "get_arrow_readers", types.Opaque("arrow_reader")(types.voidptr)
)
_del_arrow_readers = types.ExternalFunction(
    "del_arrow_readers", types.void(types.Opaque("arrow_reader"))
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


@infer_global(read_parquet_str_parallel)
class ReadParquetStrParallelInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(string_array_type, *unliteral_all(args))


@infer_global(read_parquet_list_str)
class ReadParquetListStrInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        return signature(list_string_array_type, *unliteral_all(args))


@infer_global(read_parquet_list_str_parallel)
class ReadParquetListStrParallelInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(list_string_array_type, *unliteral_all(args))


@infer_global(read_parquet_parallel)
class ReadParallelParquetInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *unliteral_all(args))


from numba.core import cgutils
from numba.core.imputils import lower_builtin
from numba.np.arrayobj import make_array
from llvmlite import ir as lir
import llvmlite.binding as ll

from bodo.config import _has_pyarrow

if _has_pyarrow:
    from bodo.io import parquet_cpp

    ll.add_symbol("get_arrow_readers", parquet_cpp.get_arrow_readers)
    ll.add_symbol("del_arrow_readers", parquet_cpp.del_arrow_readers)
    ll.add_symbol("pq_read", parquet_cpp.read)
    ll.add_symbol("pq_read_parallel", parquet_cpp.read_parallel)
    ll.add_symbol("pq_get_size", parquet_cpp.get_size)
    ll.add_symbol("pq_read_string", parquet_cpp.read_string)
    ll.add_symbol("pq_read_string_parallel", parquet_cpp.read_string_parallel)
    ll.add_symbol("pq_read_list_string", parquet_cpp.read_list_string)
    ll.add_symbol("pq_read_list_string_parallel", parquet_cpp.read_list_string_parallel)
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


@lower_builtin(
    read_parquet_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    types.Array,
    types.int32,
    types.intp,
    types.intp,
)
def pq_read_parallel_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(64),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
        ],
    )
    out_array = make_array(sig.args[2])(context, builder, args[2])
    zero_ptr = context.get_constant_null(types.voidptr)

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_parallel")
    return builder.call(
        fn,
        [
            args[0],
            args[1],
            builder.bitcast(out_array.data, lir.IntType(8).as_pointer()),
            args[3],
            args[4],
            args[5],
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


@lower_builtin(
    read_parquet_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    IntegerArrayType,
    types.int32,
    types.intp,
    types.intp,
)
@lower_builtin(
    read_parquet_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    BooleanArrayType,
    types.int32,
    types.intp,
    types.intp,
)
@lower_builtin(
    read_parquet_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    DecimalArrayType,
    types.int32,
    types.intp,
    types.intp,
)
@lower_builtin(
    read_parquet_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    datetime_date_array_type,
    types.int32,
    types.intp,
    types.intp,
)
def pq_read_parallel_int_arr_lower(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
            lir.IntType(64),
            lir.IntType(64),
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

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_parallel")
    return builder.call(
        fn,
        [
            args[0],
            args[1],
            builder.bitcast(data_array.data, lir.IntType(8).as_pointer()),
            args[3],
            args[4],
            args[5],
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


@lower_builtin(
    read_parquet_str_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.intp,
)
def pq_read_string_parallel_lower(context, builder, sig, args):
    typ = sig.return_type
    dtype = StringArrayPayloadType()
    meminfo, meminfo_data_ptr = construct_string_array(context, builder)
    str_arr_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    string_array = context.make_helper(builder, typ)
    str_arr_payload.num_strings = args[3]

    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
        ],
    )

    fn = builder.module.get_or_insert_function(fnty, name="pq_read_string_parallel")
    res = builder.call(
        fn,
        [
            args[0],
            args[1],
            str_arr_payload._get_ptr_by_name("offsets"),
            str_arr_payload._get_ptr_by_name("data"),
            str_arr_payload._get_ptr_by_name("null_bitmap"),
            args[2],
            args[3],
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


@lower_builtin(
    read_parquet_list_str_parallel,
    types.Opaque("arrow_reader"),
    types.intp,
    types.intp,
    types.intp,
)
def pq_read_list_string_parallel_lower(context, builder, sig, args):
    typ = sig.return_type
    dtype = ListStringArrayPayloadType()
    meminfo, meminfo_data_ptr = construct_list_string_array(context, builder)
    list_str_arr_payload = cgutils.create_struct_proxy(dtype)(context, builder)
    list_str_array = context.make_helper(builder, typ)
    list_str_array.num_items = args[3]

    fnty = lir.FunctionType(
        lir.IntType(32),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(64),
            lir.IntType(64),
        ],
    )

    fn = builder.module.get_or_insert_function(
        fnty, name="pq_read_list_string_parallel"
    )
    _res = builder.call(
        fn,
        [
            args[0],
            args[1],
            list_str_arr_payload._get_ptr_by_name("data_offsets"),
            list_str_arr_payload._get_ptr_by_name("index_offsets"),
            list_str_arr_payload._get_ptr_by_name("data"),
            list_str_arr_payload._get_ptr_by_name("null_bitmap"),
            args[2],
            args[3],
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
    metadata_t,
    compression_t,
    is_parallel_t,
    write_index,
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
