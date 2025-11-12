from __future__ import annotations

import json

from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.core.typing import signature
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.utils.typing import (
    get_literal_value,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.bodosql_types.table_path import (
    TablePath,
    check_tablepath_constant_arguments,
    convert_tablepath_constructor_args,
)


class TablePathType(types.Type):
    """
    Internal JIT type used to hold information about an individual table
    that should be loaded from a file. The file_path is a string
    that should describe the type of file to read.
    """

    def __init__(
        self,
        file_path,
        file_type,
        conn_str,
        reorder_io,
        db_schema,
        bodo_read_as_dict,
        statistics,
    ):
        # This assumes that file_path, file_type, and conn_str
        # are validated at a previous step, either the init
        # function or in Python.
        # TODO: Replace the file_path with the schema for better caching.
        # TODO: Remove conn_str from the caching requirement?
        super().__init__(
            name=f"TablePath({file_path}, {file_type}, {conn_str}, {reorder_io}, {db_schema}, {bodo_read_as_dict}, {json.dumps(statistics)})"
        )
        # TODO: Replace with using file_path at runtime if the schema
        # is provided.
        self._file_path = file_path
        self._file_type = file_type
        self._conn_str = conn_str
        self._reorder_io = reorder_io
        self._db_schema = db_schema
        self._bodo_read_as_dict = bodo_read_as_dict
        self._statistics = statistics

    @property
    def file_path_type(self):
        """Returns the runtime type for the filepath. Used to
        simplify generating models."""
        return types.unicode_type

    @property
    def file_type_type(self):
        """Returns the runtime type for lowering the file_type
        constant. Used for boxing into Python."""
        return types.unicode_type

    @property
    def conn_str_type(self):
        """Returns the runtime type for the conn_str. Used to
        simplify generating models."""
        if self._conn_str is None:
            return types.none
        return types.unicode_type


# Enable determining the type when using TablePath as an argument
@typeof_impl.register(TablePath)
def typeof_table_path(val, c):
    return TablePathType(
        val._file_path,
        val._file_type,
        val._conn_str,
        val._reorder_io,
        val._db_schema,
        val._bodo_read_as_dict,
        val._statistics,
    )


# Define the data model for the TablePath.
@register_model(TablePathType)
class TablePathModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("file_path", fe_type.file_path_type),
            ("conn_str", fe_type.conn_str_type),
        ]
        super().__init__(dmm, fe_type, members)


# 2nd arg is used in LLVM level, 3rd arg is used in python level
make_attribute_wrapper(TablePathType, "file_path", "_file_path")
make_attribute_wrapper(TablePathType, "conn_str", "_conn_str")


# Support boxing and unboxing in case someone passes the value as an
# argument or we need to cross into objmode
@box(TablePathType)
def box_table_path(typ, val, c):
    """
    Box a table path into a Python object. We populate
    the file_type based on typing information.
    """
    table_path = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # Load the file path from the model. This is done because we eventually want to support
    # variable paths with supported schemas.
    c.context.nrt.incref(c.builder, typ.file_path_type, table_path.file_path)
    file_path_obj = c.pyapi.from_native_value(
        typ.file_path_type, table_path.file_path, c.env_manager
    )

    file_type_obj = c.pyapi.from_native_value(
        typ.file_type_type,
        c.context.get_constant_generic(c.builder, typ.file_type_type, typ._file_type),
        c.env_manager,
    )
    # Load the conn_str from the model. This is done because we eventually want to support
    # variable paths with supported schemas.
    c.context.nrt.incref(c.builder, typ.conn_str_type, table_path.conn_str)
    conn_str_obj = c.pyapi.from_native_value(
        typ.conn_str_type, table_path.conn_str, c.env_manager
    )
    reorder_io_obj = c.pyapi.from_native_value(
        types.bool_, c.context.get_constant(types.bool_, typ._reorder_io), c.env_manager
    )

    bodo_read_as_dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(typ._bodo_read_as_dict)
    )

    # TODO Add support for passing back the statistics.

    table_path_obj = c.pyapi.unserialize(c.pyapi.serialize_object(TablePath))
    args = c.pyapi.tuple_pack([file_path_obj, file_type_obj])
    kws = c.pyapi.dict_pack(
        [
            ("conn_str", conn_str_obj),
            ("reorder_io", reorder_io_obj),
            ("bodo_read_as_dict", bodo_read_as_dict_obj),
        ]
    )
    res = c.pyapi.call(
        table_path_obj,
        args=args,
        kws=kws,
    )
    c.pyapi.decref(file_path_obj)
    c.pyapi.decref(file_type_obj)
    c.pyapi.decref(conn_str_obj)
    c.pyapi.decref(reorder_io_obj)
    c.pyapi.decref(bodo_read_as_dict_obj)
    c.pyapi.decref(table_path_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(TablePathType)
def unbox_table_path(typ, val, c):
    """
    Unbox a table path Python object into its native representation.
    We only need the information that is used at runtime.
    """
    file_path_obj = c.pyapi.object_getattr_string(val, "_file_path")
    file_path = c.pyapi.to_native_value(typ.file_path_type, file_path_obj).value

    conn_str_obj = c.pyapi.object_getattr_string(val, "_conn_str")
    conn_str = c.pyapi.to_native_value(typ.conn_str_type, conn_str_obj).value

    table_path = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    table_path.file_path = file_path
    table_path.conn_str = conn_str

    c.pyapi.decref(file_path_obj)
    c.pyapi.decref(conn_str_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())

    # _getvalue(): Load and return the value of the underlying LLVM structure.
    return NativeValue(table_path._getvalue(), is_error=is_error)


# Implement the constructor so the same code can be run in Python and JIT
@overload(TablePath, no_unliteral=True)
def overload_table_path_constructor(
    file_path,
    file_type,
    conn_str=None,
    reorder_io=None,
    db_schema=None,
    bodo_read_as_dict=None,
    statistics_file=None,
):
    """
    Table Path Constructor to enable calling TablePath("myfile", "parquet")
    directly inside JIT code.
    """

    def impl(
        file_path,
        file_type,
        conn_str=None,
        reorder_io=None,
        db_schema=None,
        bodo_read_as_dict=None,
        # TODO Add support for passing this argument.
        statistics_file=None,
    ):  # pragma: no cover
        return init_table_path(
            file_path,
            file_type,
            conn_str,
            reorder_io,
            db_schema,
            bodo_read_as_dict,
        )

    return impl


@intrinsic(prefer_literal=True)
def init_table_path(
    typingctx,
    file_path_typ,
    file_type_typ,
    conn_str_typ,
    reorder_io_typ,
    db_schema,
    bodo_read_as_dict_typ,
):
    """
    Intrinsic used to actually construct the TablePath from the constructor.
    """
    # Check for literals
    if not is_overload_constant_str(file_path_typ):
        raise_bodo_error("bodosql.TablePath(): 'file_path' must be a constant string")
    if not is_overload_constant_str(file_type_typ):
        raise_bodo_error("bodosql.TablePath(): 'file_type' must be a constant string")
    if not (is_overload_none(conn_str_typ) or is_overload_constant_str(conn_str_typ)):
        raise_bodo_error(
            "bodosql.TablePath(): `conn_str` must be a constant string if provided"
        )
    if not (
        is_overload_none(reorder_io_typ) or is_overload_constant_bool(reorder_io_typ)
    ):
        raise_bodo_error(
            "bodosql.TablePath(): `reorder_io` must be a constant boolean."
        )
    if not (is_overload_none(db_schema) or is_overload_constant_str(db_schema)):
        raise_bodo_error(
            "bodosql.TablePath(): `db_schema` must be a constant string if provided."
        )
    if not (
        is_overload_none(bodo_read_as_dict_typ)
        or is_overload_constant_list(bodo_read_as_dict_typ)
    ):
        raise_bodo_error(
            "bodosql.TablePath(): `_bodo_read_as_dict_typ` must be a constant list of strings if provided."
        )

    # Extract the literal values
    literal_file_path = get_overload_const_str(file_path_typ)
    literal_file_type = get_overload_const_str(file_type_typ)
    literal_conn_str_typ = get_literal_value(conn_str_typ)
    literal_reorder_io_typ = get_literal_value(reorder_io_typ)
    literal_db_schema_typ = get_literal_value(db_schema)
    literal_bodo_read_as_dict_typ = get_literal_value(bodo_read_as_dict_typ)
    literal_statistics_file_typ = None

    # Convert the values
    (
        literal_file_path,
        literal_file_type,
        literal_conn_str_typ,
        literal_reorder_io_typ,
        literal_bodo_read_as_dict_typ,
        literal_statistics_file_typ,
    ) = convert_tablepath_constructor_args(
        literal_file_path,
        literal_file_type,
        literal_conn_str_typ,
        literal_reorder_io_typ,
        literal_bodo_read_as_dict_typ,
        literal_statistics_file_typ,
    )

    # Error checking.
    check_tablepath_constant_arguments(
        literal_file_path,
        literal_file_type,
        literal_conn_str_typ,
        literal_reorder_io_typ,
        literal_db_schema_typ,
        literal_bodo_read_as_dict_typ,
        literal_statistics_file_typ,
    )

    def codegen(context, builder, signature, args):  # pragma: no cover
        file_path, _, conn_str, _, _, _ = args
        typ = signature.return_type
        table_path = cgutils.create_struct_proxy(typ)(context, builder)
        table_path.file_path = file_path
        table_path.conn_str = conn_str
        return table_path._getvalue()

    ret_type = TablePathType(
        literal_file_path,
        literal_file_type,
        literal_conn_str_typ,
        literal_reorder_io_typ,
        literal_db_schema_typ,
        literal_bodo_read_as_dict_typ,
        {},
    )
    # Convert file_path to unicode type because we always store it as a
    # regular string.
    return (
        signature(
            ret_type,
            ret_type.file_path_type,
            file_type_typ,
            ret_type.conn_str_type,
            reorder_io_typ,
            db_schema,
            bodo_read_as_dict_typ,
        ),
        codegen,
    )


@lower_constant(TablePathType)
def lower_constant_table_path(context, builder, ty, pyval):
    """
    Support lowering a TablePath as a constant.
    """
    # We only need the file path and conn_str because
    # the file_type should be handled by typing
    file_path = context.get_constant_generic(
        builder, ty.file_path_type, pyval._file_path
    )
    conn_str = context.get_constant_generic(builder, ty.conn_str_type, pyval._conn_str)
    table_path = cgutils.create_struct_proxy(ty)(context, builder)
    table_path.file_path = file_path
    table_path.conn_str = conn_str
    return table_path._getvalue()
